using UnityEngine;
using UnityEngine.InputSystem;

public class ToggleVoronoi : MonoBehaviour
{
    [Header("What to Toggle")]
    public GameObject voronoiVisualizer;

    [Header("Keybind")]
    public Key toggleKey = Key.V; // The new Input System's dropdown!

    void Update()
    {
        // Failsafe: Ensure a keyboard is actually plugged in
        if (Keyboard.current == null) return;

        // Listens for whatever key you selected in the Inspector
        if (Keyboard.current[toggleKey].wasPressedThisFrame)
        {
            if (voronoiVisualizer != null)
            {
                voronoiVisualizer.SetActive(!voronoiVisualizer.activeSelf);
            }
        }
    }
}