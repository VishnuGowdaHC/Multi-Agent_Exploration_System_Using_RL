using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json.Linq;

[Serializable]
public class MessageEnvelope
{
    public string type;
    public int agent_id;
    public object payload;
}

public class NetClient : MonoBehaviour
{
    public static NetClient Instance { get; private set; }

    [Header("Connection Settings")]
    public string serverUri = "ws://127.0.0.1:8766";
    public float reconnectInterval = 3.0f;
    public float heartbeatInterval = 2.0f;

    private ClientWebSocket ws;
    private CancellationTokenSource cancellationTokenSource;
    private Dictionary<int, AgentController> registeredAgents = new Dictionary<int, AgentController>();
    // delete this line:
    private bool isConnected = false;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            cancellationTokenSource = new CancellationTokenSource();
            _ = ConnectToServerAsync();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public async void SendAgentStuck(int agentId, Vector3 currentPos)
    {
        Vector3 originOffset = TerrainManager.Instance?.gridOrigin?.position ?? Vector3.zero;
        var payload = new Dictionary<string, object>
        {
            { "pos", new float[] { currentPos.x - originOffset.x, currentPos.z - originOffset.z } }
        };
        await SendMessageAsync("agent_stuck", agentId, payload);
    }

    public void RegisterAgent(int agentId, AgentController controller)
    {
        if (!registeredAgents.ContainsKey(agentId))
        {
            registeredAgents[agentId] = controller;
            Debug.Log($"[NetClient] Registered agent {agentId}");
            // If already connected, send its position immediately
            if (ws != null && ws.State == WebSocketState.Open)
                SendWaypointReached(agentId, controller.transform.position);
        }
    }

    public async Task ConnectToServerAsync()
    {
        while (!cancellationTokenSource.Token.IsCancellationRequested)
        {
            try
            {
                ws?.Dispose();
                ws = new ClientWebSocket();
                Debug.Log($"[NetClient] Connecting to {serverUri} ...");
                await ws.ConnectAsync(new Uri(serverUri), cancellationTokenSource.Token);
                Debug.Log($"[NetClient] Connected! State = {ws.State}");

                isConnected = true;
                _ = ReceiveLoopAsync();
                _ = HeartbeatLoopAsync();

                // Send initial positions for all registered agents
                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    Debug.Log($"[NetClient] Sending initial positions for {registeredAgents.Count} agents");
                    foreach (var kvp in registeredAgents)
                        SendWaypointReached(kvp.Key, kvp.Value.transform.position);
                });

                break; // exit retry loop
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[NetClient] Connection failed: {ex.Message}. Retrying in {reconnectInterval}s...");
                await Task.Delay(TimeSpan.FromSeconds(reconnectInterval), cancellationTokenSource.Token);
            }
        }
    }

    private async Task ReceiveLoopAsync()
    {
        var buffer = new byte[1024 * 64];
        Debug.Log($"[NetClient] ReceiveLoop started, State: {ws?.State}");

        while (ws != null && ws.State == WebSocketState.Open && !cancellationTokenSource.Token.IsCancellationRequested)
        {
            try
            {
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationTokenSource.Token);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closed by server", CancellationToken.None);
                    Debug.Log("[NetClient] WebSocket closed by server.");
                    isConnected = false;
                    break;
                }
                else
                {
                    string messageJson = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    Debug.Log($"[NetClient] Received: {messageJson}");
                    ProcessIncomingMessage(messageJson);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[NetClient] Receive error: {ex.Message}, State: {ws?.State}");
                isConnected = false;
                break;
            }
        }

        // If not cancelled, attempt to reconnect
        if (!cancellationTokenSource.Token.IsCancellationRequested)
        {
            Debug.Log("[NetClient] Receive loop ended – reconnecting...");
            _ = ConnectToServerAsync();
        }
    }

    private async Task HeartbeatLoopAsync()
    {
        while (ws != null && ws.State == WebSocketState.Open && !cancellationTokenSource.Token.IsCancellationRequested)
        {
            await Task.Delay(TimeSpan.FromSeconds(heartbeatInterval), cancellationTokenSource.Token);
            if (ws?.State == WebSocketState.Open)
            {
                try
                {
                    // Send a simple ping (text message) to keep the connection alive
                    var envelope = new { type = "heartbeat", agent_id = "system", payload = new { } };
                    string json = Newtonsoft.Json.JsonConvert.SerializeObject(envelope);
                    await ws.SendAsync(
                        new ArraySegment<byte>(Encoding.UTF8.GetBytes(json)),
                        WebSocketMessageType.Text,
                        true,
                        cancellationTokenSource.Token
                    );
                }
                catch { /* ignore */ }
            }
        }
    }

    private void ProcessIncomingMessage(string jsonString)
    {
        try
        {
            JObject data = JObject.Parse(jsonString);
            string type = data["type"]?.ToString();
            JToken payload = data["payload"];

            if (type == "occupancy_update")
            {
                var cells = payload["cells"]?.ToObject<List<JObject>>();
                if (cells != null)
                {
                    foreach (var cell in cells)
                    {
                        int x = cell["x"].ToObject<int>();
                        int z = cell["z"].ToObject<int>();
                        int state = cell["state"].ToObject<int>();
                        bool isExplored = state == 1;
                        bool isObstacle = state == -1;

                        UnityMainThreadDispatcher.Instance().Enqueue(() =>
                        {
                            if (TerrainManager.Instance != null)
                                TerrainManager.Instance.UpdateCellState(x, z, isExplored, isObstacle);
                        });
                    }
                }
                return;
            }

            if (data["agent_id"] == null || data["agent_id"].Type != JTokenType.Integer)
                return;

            int agentId = data["agent_id"].ToObject<int>();

            

            if (!registeredAgents.TryGetValue(agentId, out AgentController agent))
            {
                Debug.LogWarning($"[NetClient] No agent registered for ID {agentId}");
                return;
            }

            if (type == "waypoint_list")
            {
                JArray waypointArray = payload as JArray;
                List<Vector3> parsedWaypoints = new List<Vector3>();
                if (waypointArray != null)
                {
                    foreach (JToken wp in waypointArray)
                        parsedWaypoints.Add(new Vector3(wp["x"].ToObject<float>(), 0f, wp["z"].ToObject<float>()));
                }
                Debug.Log($"[NetClient] Received waypoint_list for agent {agentId}, count: {parsedWaypoints.Count}");
                UnityMainThreadDispatcher.Instance().Enqueue(() => agent.SetWaypoints(parsedWaypoints));
            }
            else if (type == "hold")
            {
                Debug.Log($"[NetClient] Received hold for agent {agentId}");
                UnityMainThreadDispatcher.Instance().Enqueue(() => agent.TriggerHold());
            }
            if (type == "voronoi_sync")
            {
                var cells = payload["cells"]?.ToObject<List<JObject>>();
                if (cells != null)
                {
                    List<Vector2> parsedCells = new List<Vector2>();
                    foreach (var cell in cells)
                    {
                        parsedCells.Add(new Vector2(cell["x"].ToObject<float>(), cell["z"].ToObject<float>()));
                    }

                    UnityMainThreadDispatcher.Instance().Enqueue(() =>
                    {
                        if (VoronoiVisualizer.Instance != null && TerrainManager.Instance != null)
                        {
                            VoronoiVisualizer.Instance.UpdateAgentZone(agentId, parsedCells, TerrainManager.Instance.gridOrigin.position);
                        }
                    });
                }
                return;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[NetClient] Failed to parse JSON: {jsonString} | Error: {ex.Message}");
        }
    }

    // --- Outbound messages ---

    public async void SendWaypointReached(int agentId, Vector3 currentPos)
    {
        Vector3 originOffset = TerrainManager.Instance?.gridOrigin?.position ?? Vector3.zero;
        float[] posArray = new float[] { currentPos.x - originOffset.x, currentPos.z - originOffset.z };
        var payload = new Dictionary<string, object> { { "pos", posArray } };
        Debug.Log($"[NetClient] SendWaypointReached agent {agentId}: pos=({posArray[0]}, {posArray[1]})");
        await SendMessageAsync("waypoint_reached", agentId, payload);
    }

    public async void SendSensorDetection(int agentId, string tag, float distance, Vector3 hazardPos)
    {
        var payload = new Dictionary<string, object>
        {
            { "tags", tag },
            { "distance", distance },
            { "hazard_pos", new float[] { hazardPos.x, hazardPos.z } }
        };
        await SendMessageAsync("sensor_detection", agentId, payload);
    }

    public async void SendObstacleBlocked(int agentId, Vector3 obstaclePos, Vector3 obstacleSize)
    {
        var payload = new Dictionary<string, object>
        {
            { "obstacle_pos", new float[] { obstaclePos.x, obstaclePos.z } },
            { "size", new float[] { obstacleSize.x, obstacleSize.z } }
        };
        await SendMessageAsync("obstacle_blocked", agentId, payload);
    }

    private async Task SendMessageAsync(string type, int agentId, object payload)
    {
        if (ws == null || ws.State != WebSocketState.Open)
        {
            Debug.LogWarning($"[NetClient] WebSocket not connected – cannot send [{type}]");
            return;
        }

        try
        {
            var envelope = new { type = type, agent_id = agentId, payload = payload };
            string json = Newtonsoft.Json.JsonConvert.SerializeObject(envelope);
            byte[] encodedBytes = Encoding.UTF8.GetBytes(json);

            await ws.SendAsync(new ArraySegment<byte>(encodedBytes), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
            Debug.Log($"[NetClient] Sent [{type}] for agent {agentId}");
            TelemetryLogger.Instance?.LogRaw($"sent_{type}", $"agent={agentId}");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[NetClient] Failed sending [{type}] : {ex.Message}");
        }
    }

    void OnDestroy()
    {
        cancellationTokenSource?.Cancel();
        ws?.Dispose();
    }
}