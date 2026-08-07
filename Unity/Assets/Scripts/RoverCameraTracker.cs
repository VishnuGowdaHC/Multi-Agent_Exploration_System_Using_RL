using UnityEngine;
using UnityEngine.InputSystem; // You MUST add this namespace for the new system

public class RoverCameraTracker : MonoBehaviour
{
    [Header("Tracking Targets")]
    [Tooltip("Drag your 4 Rover GameObjects here in order (0, 1, 2, 3)")]
    public Transform[] rovers;

    [Header("Camera Positioning")]
    public Vector3 offset = new Vector3(0, 5f, -6f); // 5 units up, 6 units back
    public float smoothSpeed = 5f;

    private int currentTargetIndex = 0;

    void LateUpdate()
    {
        if (rovers == null || rovers.Length == 0) return;

        // Switch targets using number keys 1-4 via the New Input System
        if (Keyboard.current != null)
        {
            if (Keyboard.current.digit1Key.wasPressedThisFrame) currentTargetIndex = 0;
            if (Keyboard.current.digit2Key.wasPressedThisFrame) currentTargetIndex = 1;
            if (Keyboard.current.digit3Key.wasPressedThisFrame) currentTargetIndex = 2;
            if (Keyboard.current.digit4Key.wasPressedThisFrame) currentTargetIndex = 3;
        }

        // Ensure we don't go out of bounds if fewer than 4 rovers are assigned
        currentTargetIndex = currentTargetIndex % rovers.Length;
        Transform target = rovers[currentTargetIndex];

        if (target != null)
        {
            // Calculate the position behind the rover based on its current rotation
            Vector3 desiredPosition = target.position + target.rotation * offset;
            
            // Smoothly glide to the new position
            transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);

            // Always look slightly above the center of the rover
            transform.LookAt(target.position + Vector3.up * 1f);
        }
    }
}