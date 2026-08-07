using UnityEngine;

public class FogOfWarManager : MonoBehaviour
{
    public static FogOfWarManager Instance { get; private set; }

    [Header("Map Settings")]
    public int mapWidth = 30;
    public int mapHeight = 30;

    [Header("Behavior")]
    [Tooltip("If true, previously revealed areas fade back into smoke once the rover moves away.")]
    public bool smokeRegrows = true;
    [Range(0f, 2f)] public float regrowSpeed = 0.15f;
    [Range(0.5f, 10f)] public float revealSpeed = 6f;

    [Header("Visuals")]
    public Material fogMaterial;

    private Texture2D visionMaskTexture;
    private float[,] visibility;
    private byte[] pixelBuffer;
    private bool dirty;

    void Awake()
    {
        Instance = this;
        InitializeFog();
    }

    private void InitializeFog()
    {
        visionMaskTexture = new Texture2D(mapWidth, mapHeight, TextureFormat.R8, false, true);
        visionMaskTexture.filterMode = FilterMode.Bilinear;
        visionMaskTexture.wrapMode = TextureWrapMode.Clamp;

        visibility = new float[mapWidth, mapHeight];
        pixelBuffer = new byte[mapWidth * mapHeight];

        PushToTexture();
        if (fogMaterial != null)
            fogMaterial.SetTexture("_VisionMask", visionMaskTexture);
    }

    /// <summary>
    /// Called every frame by each RoverVision in range — pushes visibility
    /// toward 1 (clear) for cells within radius, with soft distance falloff.
    /// </summary>
    public void RevealRadius(Vector3 worldPos, float radius)
    {
        if (TerrainManager.Instance == null || TerrainManager.Instance.gridOrigin == null) return;

        Vector3 origin = TerrainManager.Instance.gridOrigin.position;
        int cx = Mathf.RoundToInt(worldPos.x - origin.x);
        int cz = Mathf.RoundToInt(worldPos.z - origin.z);
        int r = Mathf.CeilToInt(radius) + 1; // pad for soft edge

        for (int x = cx - r; x <= cx + r; x++)
        {
            for (int z = cz - r; z <= cz + r; z++)
            {
                if (x < 0 || x >= mapWidth || z < 0 || z >= mapHeight) continue;

                float dist = Mathf.Sqrt((x - cx) * (x - cx) + (z - cz) * (z - cz));
                if (dist > radius) continue;

                float falloff = 1f - Mathf.Clamp01(dist / radius); // stronger near center
                float pull = Mathf.Max(falloff, 0.25f) * revealSpeed * Time.deltaTime;

                visibility[x, z] = Mathf.Max(visibility[x, z], Mathf.MoveTowards(visibility[x, z], 1f, pull));
                dirty = true;
            }
        }
    }

    void LateUpdate()
    {
        // Runs after all RoverVision.Update() reveal calls this frame
        if (smokeRegrows)
        {
            for (int x = 0; x < mapWidth; x++)
            {
                for (int z = 0; z < mapHeight; z++)
                {
                    if (visibility[x, z] > 0f)
                    {
                        visibility[x, z] = Mathf.MoveTowards(visibility[x, z], 0f, regrowSpeed * Time.deltaTime);
                        dirty = true;
                    }
                }
            }
        }

        if (dirty)
        {
            PushToTexture();
            dirty = false;
        }
    }

    private void PushToTexture()
    {
        for (int x = 0; x < mapWidth; x++)
            for (int z = 0; z < mapHeight; z++)
                pixelBuffer[z * mapWidth + x] = (byte)(Mathf.Clamp01(visibility[x, z]) * 255);

        visionMaskTexture.SetPixelData(pixelBuffer, 0);
        visionMaskTexture.Apply(false);
    }

    public float GetVisibilityAt(Vector3 worldPos)
    {
        if (TerrainManager.Instance == null || TerrainManager.Instance.gridOrigin == null) return 0f;

        Vector3 origin = TerrainManager.Instance.gridOrigin.position;
        int x = Mathf.RoundToInt(worldPos.x - origin.x);
        int z = Mathf.RoundToInt(worldPos.z - origin.z);

        if (x < 0 || x >= mapWidth || z < 0 || z >= mapHeight) return 0f;
        
        // Returns a value between 0f (total smoke) and 1f (totally clear)
        return visibility[x, z]; 
    }
}