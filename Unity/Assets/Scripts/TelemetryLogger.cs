using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text;
using UnityEngine;

public class TelemetryLogger : MonoBehaviour
{
    public static TelemetryLogger Instance { get; private set; }

    private ConcurrentQueue<string> logQueue = new ConcurrentQueue<string>();
    private StreamWriter writer;
    private string filePath;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);

            filePath = Path.Combine(Application.persistentDataPath, $"telemetry_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
            writer = new StreamWriter(filePath, append: false, Encoding.UTF8);
            writer.WriteLine("timestamp,agent_id,event,x,z,extra");
            writer.Flush();

            Debug.Log($"[TelemetryLogger] Writing to {filePath}");
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void Log(int agentId, string eventType, float x, float z, string extra = "")
    {
        string line = $"{Time.time:F3},{agentId},{eventType},{x:F4},{z:F4},{extra}";
        logQueue.Enqueue(line);
    }

    public void LogRaw(string eventType, string extra = "")
    {
        string line = $"{Time.time:F3},-1,{eventType},,,{extra}";
        logQueue.Enqueue(line);
    }

    void Update()
    {
        // Flush queued entries on main thread each frame
        while (logQueue.TryDequeue(out string line))
        {
            writer.WriteLine(line);
        }
    }

    void OnApplicationQuit()
    {
        writer?.Flush();
        writer?.Close();
        Debug.Log($"[TelemetryLogger] Saved to {filePath}");
    }
}