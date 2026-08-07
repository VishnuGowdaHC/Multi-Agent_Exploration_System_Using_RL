using UnityEngine;

public class HazardVisualizer : MonoBehaviour
{
    [Header("Hazard Settings")]
    [Tooltip("Should match the radius_m value for this hazard in jungle_demo.yml")]
    public float dangerRadius = 6f;

    [Header("Visuals")]
    [ColorUsage(true, true)] // CRUCIAL: The second 'true' enables HDR, allowing actual emission/glow!
    public Color zoneColor = new Color(1f, 0f, 0.1f, 1f); 
    
    [Tooltip("Makes the hazard pulse menacingly")]
    public bool pulseEffect = true;
    public float pulseSpeed = 2f;
    public float pulseMinAlpha = 0.2f;
    public float pulseMaxAlpha = 0.8f;

    private Material hazardMaterial;
    private Color baseGlowColor;

    void Start()
    {
        GameObject zone = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        zone.name = "DangerZoneVisual";
        zone.transform.SetParent(transform, false);
        
        // Flattened it to look like a projection decal, lifted slightly above ground
        zone.transform.localPosition = new Vector3(0f, 0.05f, 0f); 
        zone.transform.localScale = new Vector3(dangerRadius * 2f, 0.001f, dangerRadius * 2f);

        Destroy(zone.GetComponent<Collider>()); // visual only, keep sensors clear

        // URP Particles Unlit is built natively to handle alpha blending and color tinting
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
        if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");

        hazardMaterial = new Material(shader);

        // Inject our code-generated soft ring texture
        hazardMaterial.mainTexture = GenerateRingTexture(256);

        baseGlowColor = zoneColor;
        hazardMaterial.SetColor("_BaseColor", baseGlowColor);

        // Force URP into Transparent Alpha mode
        hazardMaterial.SetFloat("_Surface", 1f);
        hazardMaterial.SetFloat("_Blend", 0f); 
        hazardMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        hazardMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        hazardMaterial.SetInt("_ZWrite", 0);
        hazardMaterial.renderQueue = 3000;

        zone.GetComponent<Renderer>().material = hazardMaterial;
    }

    void Update()
    {
        if (pulseEffect && hazardMaterial != null)
        {
            // Smoothly throb the alpha channel over time
            float pingPong = Mathf.PingPong(Time.time * pulseSpeed, 1f);
            float currentAlpha = Mathf.Lerp(pulseMinAlpha, pulseMaxAlpha, pingPong);

            Color pulseColor = baseGlowColor;
            pulseColor.a = currentAlpha;

            hazardMaterial.SetColor("_BaseColor", pulseColor);
        }
    }

    /// <summary>
    /// Mathematically draws a soft warning ring so it doesn't look like a solid block
    /// </summary>
    private Texture2D GenerateRingTexture(int resolution)
    {
        Texture2D tex = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
        Vector2 center = new Vector2(resolution / 2f, resolution / 2f);
        float radius = resolution / 2f;

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                float dist = Vector2.Distance(new Vector2(x, y), center);
                float normalizedDist = dist / radius;

                float alpha = 0f;
                if (normalizedDist <= 1f)
                {
                    // Push opacity to the outer edge (creates an empty center)
                    alpha = Mathf.Pow(normalizedDist, 4f);

                    // Soften the absolute outer edge so it doesn't pixelate
                    float outerEdgeFade = Mathf.Clamp01((1f - normalizedDist) * 15f);
                    alpha *= outerEdgeFade;
                }

                // Apply white with our calculated alpha (the Material's _BaseColor will tint it)
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        tex.Apply();
        return tex;
    }
}