using UnityEngine;

public class TerrainManager : MonoBehaviour
{
    public static TerrainManager Instance { get; private set; }

    [Header("Grid Display Settings")]
    public int gridWidth = 30; // Make sure this matches Python's 30x30 grid!
    public int gridHeight = 30;
    public float cellSize = 1.0f;

    [Header("Visual Prefabs")]
    // Removed unexploredFogPrefab — we use FogOfWarManager now!
    public GameObject obstaclePrefab;

    private GameObject[,] obstacleGrid;

    [Header("Grid Origin")]
    public Transform gridOrigin;

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    void Start()
    {
        InitializeTerrainGrid();
    }

    private void InitializeTerrainGrid()
    {
        // Only initialize the obstacle array. 
        // The heavy massive pillar instantiation loop is completely deleted!
        obstacleGrid = new GameObject[gridWidth, gridHeight];
    }

    /// <summary>
    /// Called when the Python backend pushes down occupancy or exploration updates 
    /// to update the fog-of-war or terrain state visually.
    /// </summary>
    public void UpdateCellState(int x, int z, bool isExplored, bool isObstacle)
    {
        if (x < 0 || x >= gridWidth || z < 0 || z >= gridHeight)
        {
            Debug.LogWarning($"[TerrainManager] Cell state update out of bounds: ({x},{z})");
            return;
        }
        
        Vector3 worldPos = gridOrigin.position + new Vector3(x * cellSize, 0f, z * cellSize);

        
        // Place obstacle prefab if this cell is impassable and doesn't already have one
        if (isObstacle && obstacleGrid[x, z] == null && obstaclePrefab != null)
        {
            obstacleGrid[x, z] = Instantiate(obstaclePrefab, worldPos, Quaternion.identity, transform);
        }
    }
}