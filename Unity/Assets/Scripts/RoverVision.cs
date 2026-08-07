using UnityEngine;

public class RoverVision : MonoBehaviour
{
    [Header("Vision Settings")]
    public float visionRadius = 4.0f; // How far the rover can see through the fog

    void Update()
    {
        // As long as the fog manager exists, continuously clear the air around the rover
        if (FogOfWarManager.Instance != null)
        {
            FogOfWarManager.Instance.RevealRadius(transform.position, visionRadius);
        }
    }
}