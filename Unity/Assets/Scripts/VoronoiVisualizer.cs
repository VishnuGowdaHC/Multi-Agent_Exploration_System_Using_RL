using System.Collections.Generic;
using UnityEngine;

public class VoronoiVisualizer : MonoBehaviour
{
    public static VoronoiVisualizer Instance { get; private set; }

    [Header("Visualization Settings")]
    public GameObject tilePrefab; // A simple flat Quad or Cube
    public LayerMask groundLayer; // Set this to hit your Ground_03 mesh
    public float raycastHeight = 50f;
    public float gridResolution = 1.0f; // Matches your Python backend

    [Header("Agent Colors")]
    public Material[] agentMaterials; 

    // Tracks instantiated tiles by Agent ID so we can clear them on updates
    private Dictionary<int, List<GameObject>> activeTiles = new Dictionary<int, List<GameObject>>();

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void UpdateAgentZone(int agentId, List<Vector2> gridCells, Vector3 gridOrigin)
    {
        Debug.Log($"[Voronoi] Update triggered for Agent {agentId}. Received {gridCells.Count} cells from Python.");

        // Clear old tiles for this agent
        if (activeTiles.ContainsKey(agentId))
        {
            foreach (var tile in activeTiles[agentId]) Destroy(tile);
            activeTiles[agentId].Clear();
        }
        else
        {
            activeTiles[agentId] = new List<GameObject>();
        }

        Material agentMat = agentMaterials[agentId % agentMaterials.Length];
        int missCount = 0; 

        foreach (Vector2 cell in gridCells)
        {
            float worldX = gridOrigin.x + (cell.x * gridResolution) + (gridResolution / 2f);
            float worldZ = gridOrigin.z + (cell.y * gridResolution) + (gridResolution / 2f);

            Vector3 rayStart = new Vector3(worldX, raycastHeight, worldZ);
            
            // X-RAY VISION: Draws a visible red laser in the Scene view for 5 seconds
            Debug.DrawRay(rayStart, Vector3.down * 100f, Color.red, 5f);

            if (Physics.Raycast(rayStart, Vector3.down, out RaycastHit hit, 100f, groundLayer))
            {
                // Lifted to 0.5f to guarantee it clears the fog Plane!
                Vector3 spawnPos = hit.point + new Vector3(0, 0.5f, 0); 
                
                GameObject newTile = Instantiate(tilePrefab, spawnPos, Quaternion.Euler(90, 0, 0), transform);
                newTile.transform.up = hit.normal;
                newTile.GetComponent<Renderer>().material = agentMat;
                activeTiles[agentId].Add(newTile);
            }
            else
            {
                missCount++;
            }
        }

        if (missCount > 0)
        {
            Debug.LogWarning($"[Voronoi] Agent {agentId}: {missCount} raycasts missed the ground mesh entirely!");
        }
    }
}