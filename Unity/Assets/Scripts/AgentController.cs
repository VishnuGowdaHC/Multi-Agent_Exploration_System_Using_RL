using System.Collections.Generic;
using UnityEngine;

public class AgentController : MonoBehaviour
{
    [Header("Agent Identification")]
    public int agentId;

    [Header("Movement Settings")]
    public float moveSpeed = 3.5f;
    public float rotationSpeed = 10.0f;
    public float waypointThreshold = 0.3f;

    [Header("Physics")]
    public Rigidbody rb;

    [Header("References")]
    public Animator animator;

    // Current path received from Python
    private List<Vector3> currentWaypoints = new List<Vector3>();

    private Vector3 targetPosition;
    private bool isMoving = false;
    private bool isHolding = false;

    [Header("Stuck Detection")]
    public float stuckVelocityThreshold = 0.15f;
    public float stuckTimeThreshold = 1.5f;
    private float stuckTimer = 0f;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        }
        else
        {
            Debug.LogWarning($"[AgentController] Agent {agentId} has no Rigidbody attached — add one for physics-based movement.");
        }

        if (animator == null)
            animator = GetComponent<Animator>();

        if (NetClient.Instance != null)
            NetClient.Instance.RegisterAgent(agentId, this);

        targetPosition = transform.position;
    }

    private void Update()
    {
        UpdateAnimatorState();
    }

    private void FixedUpdate()
    {
        HandlePhysicsMovement();
    }

    /// <summary>
    /// Called by NetClient when Python sends a waypoint list.
    /// Python sends XZ only. Physics resolves the Y coordinate via gravity.
    /// </summary>
    public void SetWaypoints(List<Vector3> newWaypoints)
    {
        currentWaypoints = newWaypoints;
        isHolding = false;

        if (currentWaypoints != null && currentWaypoints.Count > 0)
        {
            targetPosition = GetTargetPosition(currentWaypoints[0]);
            isMoving = true;
        }
        else
        {
            isMoving = false;
        }
    }

    /// <summary>
    /// Called when Python sends a hold command.
    /// </summary>
    public void TriggerHold()
    {
        currentWaypoints.Clear();
        isMoving = false;
        isHolding = true;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("obstacle"))
        {
            rb.linearVelocity = Vector3.zero;
            NetClient.Instance?.SendObstacleBlocked(agentId, collision.transform.position, collision.collider.bounds.size);
        }
    }

    private void HandlePhysicsMovement()
    {
        if (!isMoving || isHolding)
        {
            rb.linearVelocity = new Vector3(0f, rb.linearVelocity.y, 0f);
            stuckTimer = 0f;
            return;
        }

        Vector3 toTarget = targetPosition - rb.position;
        toTarget.y = 0f;
        float dist = toTarget.magnitude;

        if (dist <= waypointThreshold)
        {
            stuckTimer = 0f;
            rb.linearVelocity = new Vector3(0f, rb.linearVelocity.y, 0f);

            if (currentWaypoints.Count > 0)
                currentWaypoints.RemoveAt(0);

            if (currentWaypoints.Count > 0)
            {
                targetPosition = GetTargetPosition(currentWaypoints[0]);
            }
            else
            {
                isMoving = false;
                NetClient.Instance?.SendWaypointReached(agentId, transform.position);
            }
            return;
        }

        // Stuck check: trying to move, but not actually going anywhere (pressed against something)
        Vector3 horizontalVel = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
        if (horizontalVel.magnitude < stuckVelocityThreshold)
        {
            stuckTimer += Time.fixedDeltaTime;
            if (stuckTimer >= stuckTimeThreshold)
            {
                stuckTimer = 0f;
                NetClient.Instance?.SendAgentStuck(agentId, transform.position);
            }
        }
        else
        {
            stuckTimer = 0f;
        }

        Vector3 dir = toTarget.normalized;
        rb.linearVelocity = new Vector3(dir.x * moveSpeed, rb.linearVelocity.y, dir.z * moveSpeed);

        Quaternion targetRotation = Quaternion.LookRotation(dir);
        
        rb.rotation = Quaternion.Slerp(rb.rotation, targetRotation, rotationSpeed * Time.fixedDeltaTime);
        TelemetryLogger.Instance?.Log(agentId, "position", rb.position.x, rb.position.z, $"vel={dir.magnitude * moveSpeed:F2}");
    }
    /// <summary>
    /// Converts Python XZ waypoint into a Unity world-space target.
    /// Y is intentionally left at the rover's current height — physics
    /// (gravity + ground collider) resolves vertical position every frame,
    /// so no raycast is needed here.
    /// </summary>
    private Vector3 GetTargetPosition(Vector3 waypoint)
    {
        Vector3 originOffset = TerrainManager.Instance != null && TerrainManager.Instance.gridOrigin != null
            ? TerrainManager.Instance.gridOrigin.position
            : Vector3.zero;

        float currentY = rb != null ? rb.position.y : transform.position.y;

        return new Vector3(
            originOffset.x + waypoint.x,
            currentY,
            originOffset.z + waypoint.z
        );
    }

    private void UpdateAnimatorState()
    {
        if (animator == null)
            return;

        animator.SetBool("IsExploring", isMoving && !isHolding);
        animator.SetBool("IsHolding", isHolding);
    }

    /// <summary>
    /// Used by WaypointVisualizer later.
    /// </summary>
    public List<Vector3> GetCurrentWaypoints()
    {
        return currentWaypoints;
    }

    public bool IsMoving()
    {
        return isMoving;
    }

    public bool IsHolding()
    {
        return isHolding;
    }
}