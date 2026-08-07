using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(AgentController))]
public class SensorController : MonoBehaviour
{
    [Header("Sensor Settings")]
    [Tooltip("Maximum detection radius in meters.")]
    public float detectionRadius = 8f;

    [Tooltip("Only objects on this layer will be scanned.")]
    public LayerMask detectableLayer;

    [Tooltip("Seconds between scans.")]
    public float scanInterval = 0.2f;

    [Header("Debug")]
    public bool drawDebug = true;

    private AgentController agentController;
    private int agentId;
    private float scanTimer;
    private HashSet<Collider> reportedObstacles = new HashSet<Collider>();
    private void Start()
    {
        agentController = GetComponent<AgentController>();
        agentId = agentController.agentId;

        scanTimer = scanInterval;
    }

    private void Update()
    {
        scanTimer -= Time.deltaTime;

        if (scanTimer <= 0f)
        {
            ScanEnvironment();
            scanTimer = scanInterval;
        }
    }

    private void ScanEnvironment()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, detectionRadius, detectableLayer);
        
        // Grab the exact same gridOrigin you use for the rovers
        Vector3 originOffset = TerrainManager.Instance?.gridOrigin?.position ?? Vector3.zero;

        foreach (Collider hit in hits)
        {
            if (hit.gameObject == gameObject || hit.CompareTag("agent"))
                continue;

            string objectTag = hit.tag;
            string lowerTag = objectTag.ToLower();
            
            // THE FIX: Apply the exact same offset subtraction used in NetClient.cs!
            // This perfectly translates the Hazard into the positive 0-30 grid space.
            Vector3 localHazardPosition = new Vector3(
                hit.transform.position.x - originOffset.x,
                0f,
                hit.transform.position.z - originOffset.z
            );
            
            float distance = Vector3.Distance(transform.position, hit.transform.position);

            if (NetClient.Instance != null)
            {
                if (lowerTag == "obstacle")
                {
                   if (!reportedObstacles.Contains(hit))
                    {
                        reportedObstacles.Add(hit);
                        NetClient.Instance.SendObstacleBlocked(agentId, localHazardPosition, hit.bounds.size);
                    }
                }
                else if (lowerTag == "wolf" || lowerTag == "wasp" || lowerTag == "cow")
                {
                    NetClient.Instance.SendSensorDetection(
                        agentId, lowerTag, distance, localHazardPosition
                    );
                }
            }
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (!drawDebug)
            return;

        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(
            transform.position,
            detectionRadius
        );
    }
}