using UnityEngine;

[RequireComponent(typeof(Light))]
public class TorchFlicker : MonoBehaviour
{
    [Header("Intensity")]
    public float baseIntensity = 3.5f;
    public float flickerAmount = 0.6f;
    public float flickerSpeed = 8f;

    [Header("Range (optional, subtler)")]
    public bool flickerRange = true;
    public float baseRange = 8f;
    public float rangeFlickerAmount = 0.4f;

    [Header("Color Drift (optional)")]
    public bool flickerColor = true;
    public Color warmColor = new Color(1f, 0.55f, 0.2f);
    public Color coolColor = new Color(1f, 0.65f, 0.35f);

    [Header("Randomness")]
    [Tooltip("Higher = choppier, more erratic flicker. Lower = smooth breathing glow.")]
    public float noiseScale = 1.3f;

    private Light torchLight;
    private float seed;

    void Awake()
    {
        torchLight = GetComponent<Light>();
        // Random offset so multiple torches don't flicker in sync
        seed = Random.Range(0f, 1000f);
    }

    void Update()
    {
        float t = (Time.time + seed) * flickerSpeed;

        // Layered Perlin noise for organic, non-repeating flicker
        float n1 = Mathf.PerlinNoise(t * noiseScale, 0f);
        float n2 = Mathf.PerlinNoise(t * noiseScale * 2.7f, 100f);
        float noise = (n1 * 0.7f + n2 * 0.3f) * 2f - 1f; // -1..1

        torchLight.intensity = baseIntensity + noise * flickerAmount;

        if (flickerRange)
        {
            float rangeNoise = Mathf.PerlinNoise(t * noiseScale * 0.5f, 50f) * 2f - 1f;
            torchLight.range = baseRange + rangeNoise * rangeFlickerAmount;
        }

        if (flickerColor)
        {
            float colorNoise = Mathf.PerlinNoise(t * noiseScale * 0.3f, 200f);
            torchLight.color = Color.Lerp(coolColor, warmColor, colorNoise);
        }
    }
}