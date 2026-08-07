using UnityEngine;

public class WheelController : MonoBehaviour
{
    [Header("Wheel References")]
    public Transform[] wheels;

    [Header("Settings")]
    [Tooltip("Radius of wheel in Unity units (meters).")]
    public float wheelRadius = 0.18f;

    [Tooltip("Reverse rotation direction if needed.")]
    public bool invertRotation = false;

    private Vector3 lastPosition;

    private void Start()
    {
        lastPosition = transform.position;
    }

    private void LateUpdate()
    {
        RotateWheels();
    }

    private void RotateWheels()
    {
        Vector3 movement = transform.position - lastPosition;

        if (movement.sqrMagnitude < 0.000001f)
            return;

        float distance = movement.magnitude;

        float angle =
            (distance / (2f * Mathf.PI * wheelRadius)) * 360f;

        if (invertRotation)
            angle = -angle;

        foreach (Transform wheel in wheels)
        {
            if (wheel != null)
            {
                wheel.Rotate(Vector3.right, angle, Space.Self);
            }
        }

        lastPosition = transform.position;
    }
}