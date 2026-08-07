using UnityEngine;
using UnityEngine.AI;

public class EntityBehavior : MonoBehaviour
{
    [Header("Behavior Settings")]
    public float wanderRadius = 10f;
    public float wanderTimer = 5f;
    public float movementSpeed = 2.5f;
    public string persistence = "static";

    [Header("Animation")]
    public Animator animator;

    private NavMeshAgent navAgent;
    private float timer;

    void Start()
    {
        navAgent = GetComponent<NavMeshAgent>();
        if (navAgent != null)
        {
            navAgent.speed = movementSpeed;
        }

        // Auto-fetch animator if you forgot to drag it in the inspector
        if (animator == null)
        {
            animator = GetComponentInChildren<Animator>();
        }

        timer = wanderTimer;
    }

    void Update()
    {
        // 1. Handle Movement (Only if dynamic)
        if (persistence == "dynamic")
        {
            timer += Time.deltaTime;

            if (timer >= wanderTimer)
            {
                Vector3 newPos = RandomNavSphere(transform.position, wanderRadius, -1);
                if (navAgent != null)
                {
                    navAgent.SetDestination(newPos);
                }
                timer = 0f;
            }
        }

        // 2. Handle Animation (Runs every frame)
        UpdateAnimations();
    }

    private void UpdateAnimations()
    {
        if (animator == null) return;

        // Calculate if the entity is actually physically moving across the NavMesh
        bool isMoving = false;
        if (navAgent != null && navAgent.pathPending == false)
        {
            // If the velocity is greater than a tiny threshold, it's moving
            if (navAgent.velocity.sqrMagnitude > 0.05f)
            {
                isMoving = true;
            }
        }

        // Send the state to the Animator
        animator.SetBool("IsMoving", isMoving);
    }

    public static Vector3 RandomNavSphere(Vector3 origin, float dist, int layermask)
    {
        Vector3 randDirection = Random.insideUnitSphere * dist;
        randDirection += origin;
        NavMeshHit navHit;
        NavMesh.SamplePosition(randDirection, out navHit, dist, layermask);
        return navHit.position;
    }
}