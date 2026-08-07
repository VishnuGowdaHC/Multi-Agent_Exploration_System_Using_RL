using UnityEngine;

public class HazardVisualizer : MonoBehaviour
{
    [Header("Hazard Settings")]
    [Tooltip("Should match the radius_m value for this hazard in jungle_demo.yml")]
    public float dangerRadius = 6f;

    [Header("Visuals")]
    [ColorUsage(true, true)] 
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
        GameObject zone = GameObject.CreatePrimitive(PrimitiveType.Quad);
        zone.name = "DangerZoneVisual";
        zone.transform.SetParent(transform, false);
        
        zone.transform.localPosition = new Vector3(0f, 0.05f, 0f); 
        zone.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        zone.transform.localScale = new Vector3(dangerRadius * 2f, dangerRadius * 2f, 1f);

        Destroy(zone.GetComponent<Collider>());

        // Correct URP Shaders for Unity 6
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
        if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");

        hazardMaterial = new Material(shader);
        hazardMaterial.mainTexture = GenerateRingTexture(256);

        baseGlowColor = zoneColor;
        
        // URP uses _BaseColor instead of _Color
        hazardMaterial.SetColor("_BaseColor", baseGlowColor);

        // URP Transparent Alpha Setup
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
            float pingPong = Mathf.PingPong(Time.time * pulseSpeed, 1f);
            float currentAlpha = Mathf.Lerp(pulseMinAlpha, pulseMaxAlpha, pingPong);

            Color pulseColor = baseGlowColor;
            pulseColor.a = currentAlpha;

            // URP property
            hazardMaterial.SetColor("_BaseColor", pulseColor);
        }
    }

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
                    alpha = Mathf.SmoothStep(0f, 1f, Mathf.Pow(normalizedDist, 2f));
                    float outerEdgeFade = Mathf.Clamp01((1f - normalizedDist) * 15f);
                    alpha *= outerEdgeFade;
                }

                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        tex.Apply();
        return tex;
    }
}