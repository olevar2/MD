Here are the project rules for your AI code editor, based on the Forex Trading Platform Optimization Plan:

**A. Foundational Principles & Execution**

1.  **Expert Persona:** Operate as an Expert Code and Programming Engineer. Your actions and code modifications should reflect a deep understanding of software architecture, domain-driven design, and best practices.
2.  **Plan Adherence:** Strictly follow the principles, methodologies, and architectural vision outlined in the "Forex Trading Platform Optimization Plan." Implement any new tasks or modifications stage by stage and step by step, according to the established priorities and goals of each phase (Domain-Driven Architecture, Error Handling & Resilience, Code Quality).
3.  **In-Depth Analysis:** Before implementing any changes or writing new code, conduct a thorough examination of the relevant sections of the existing codebase to understand the current state, dependencies, and potential impacts.
4.  **Platform Cleaning:** During all operations, actively identify and, after careful examination and confirmation, propose or implement the removal of harmful files, duplicate logic, dead code, or inefficient patterns to continuously improve the platform's health.
5.  **Incremental & Iterative Development:** Apply changes incrementally. For significant changes, break them down into smaller, manageable, and testable steps.
6.  **Verification:** After any modification, run all relevant tests and verify functionality in a development/staging environment to ensure correctness and prevent regressions.
7.  **Documentation:** Maintain and update documentation for any new services, interfaces, significant architectural changes, or refactoring efforts. Ensure documentation aligns with the standards set in the plan (e.g., ADRs, interface contracts).

**B. Architectural Integrity & Domain-Driven Design (Reflecting Phase 1)**

8.  **Service Boundary Clarity:**
    * Ensure every service has clear, well-documented responsibilities aligned with specific domain concepts.
    * For new functionalities or services, define boundaries based on domain analysis to prevent blurred responsibilities.
    * Uphold the separation of concerns as defined by the established service layers (Foundation, Data, Analysis, Execution, Presentation, Cross-cutting).
9.  **Domain-Driven Design (DDD) Enforcement:**
    * All new and modified code must align with DDD principles.
    * Utilize and extend the unified domain models for core concepts (e.g., ML-related, market data, trading concepts).
    * Maintain a common language for trading concepts across all services.
10. **Dependency Inversion & Interface-Based Design:**
    * Services MUST depend on abstractions (interfaces) rather than concrete implementations for inter-service communication.
    * Place all shared interfaces in `common-lib`.
    * Implement the Adapter Pattern within services to connect to these interfaces. Adapters must handle error cases gracefully and include appropriate logging/monitoring.
    * Strictly avoid direct service-to-service imports that bypass defined interfaces.
11. **Circular Dependency Prevention:**
    * Do not introduce new circular dependencies between services or modules.
    * Utilize the CI/CD pipeline's dependency analysis tools to detect and prevent cycles.
    * If a potential cycle is identified during development, re-evaluate service boundaries or apply abstraction techniques (like event-driven communication or interface adjustments) to resolve it.
12. **Service Communication Standards:**
    * Adhere to the established guidelines for service-to-service communication (e.g., event-bus for asynchronous communication, REST APIs for synchronous, defined patterns in ADRs).
    * Ensure new communication patterns are reviewed and documented.
13. **New Service Creation:** When creating new services (e.g., `market-data-service`, `order-management-service`, `notification-service` or others as needed):
    * Design them with clear domain responsibilities from the outset.
    * Implement them using the established architectural patterns (DDD, dependency inversion, standardized error handling).
    * Integrate them into the existing interface and communication structure.

**C. Resilience & Error Handling (Reflecting Phase 2)**

14. **Comprehensive Error Handling:**
    * Implement robust error handling in all new and modified code.
    * Utilize the established domain-specific exception hierarchy.
    * Employ standardized error handling middleware for all services.
15. **Resilience Pattern Implementation:**
    * Apply resilience patterns (Circuit Breaker, Retry with exponential backoff, Bulkhead, Fallback mechanisms) for cross-service communication and critical operations, as appropriate.
    * Do not over-engineer; apply these patterns judiciously where they provide clear benefits.
16. **Standardized Error Responses & Logging:**
    * Ensure all API error responses conform to the defined structure, including semantic error codes and correlation IDs for cross-service tracking.
    * Implement consistent structured logging for errors, providing sufficient contextual information for debugging.
    * Avoid swallowing exceptions without proper logging.
17. **Error Handling Coverage:** Maintain a high level of error handling coverage (target: at least 80% across all services, and higher for critical services). Proactively improve coverage in areas below this threshold.
18. **Actionable Error Messages:** Error messages must be clear and actionable for both end-users (where appropriate) and developers/support staff.
19. **Anti-Pattern Avoidance (Error Handling):**
    * Do NOT use generic `try/except` blocks that catch all exceptions without specific handling.
    * Do NOT leave error responses inconsistent across services.
    * Do NOT omit correlation IDs in error paths involving multiple services.

**D. Code Quality, Maintainability & Refactoring (Reflecting Phase 3)**

20. **Refactoring Large Files & Complex Modules:**
    * **Safety First (Strict Adherence Required for any refactoring):**
        * Create comprehensive tests (unit, integration) that cover existing behavior *before* starting any refactoring.
        * Refactor one logical component or responsibility at a time.
        * Strive to maintain the same public interfaces. If changes are necessary, use facades or compatibility layers to minimize disruption during a transition period.
        * Run the full test suite after each incremental change.
        * Document all significant refactoring changes and their rationale.
        * Verify functionality thoroughly in a development environment before merging.
    * Break down large files (>50KB as a general guideline, or any file violating SRP) into smaller, domain-focused components or modules. Apply the specific strategies detailed in Phase 3.1 (e.g., creating packages, separating concerns like calculation from analysis) for similar files.
21. **Coding Standards & Conventions:**
    * Adhere to consistent coding standards across all services and languages (Python, TypeScript React). If a specific standard is defined for the project, follow it.
    * Enforce consistent naming conventions (e.g., choose one of snake_case, camelCase for variables/functions and kebab-case for services/endpoints, and apply it uniformly). Resolve existing inconsistencies when working on affected code.
22. **Code Duplication Reduction:** Actively identify and refactor duplicated code, especially in areas like indicator implementations or common utility functions. Promote reuse through shared libraries or well-defined helper modules.
23. **Layered Architecture Adherence:** Organize code within services according to the layered architecture pattern (e.g., presentation, application, domain, infrastructure layers where applicable).
24. **Single Responsibility Principle (SRP):** Ensure that classes, functions, and modules have a single, well-defined responsibility.
25. **Testing Robustness:**
    * Write comprehensive unit, integration, and (where applicable) end-to-end tests for all new features and bug fixes.
    * Ensure all tests pass consistently.
    * Address and resolve any testing framework configuration issues (e.g., `pytest-asyncio` warnings) to maintain a stable testing environment.
    * Fix failing integration tests and configuration warnings.
26. **Anti-Pattern Avoidance (Code Quality):**
    * Avoid creating "pass-through" interfaces that merely mirror existing implementations without abstracting a true domain concept.
    * Avoid rushing to refactor service boundaries or code without proper analysis, domain understanding, and testing.
    * Prioritize maintainable architecture and code over quick fixes that just make the code compile.
    * Avoid creating overly complex interfaces or components that try to solve too many problems at once.

**E. Ongoing Maintenance & Evolution**

27. **Continuous Improvement:** Proactively identify areas for improvement in the codebase, architecture, and processes, aligning with the overall optimization plan.
28. **Prevent Regression:** Ensure that new changes do not reintroduce previously fixed issues (e.g., circular dependencies, poor error handling, inconsistent naming).
29. **Configuration with Code:** Manage configurations for services and infrastructure as code where possible, and version control them.
30. **Security Considerations:** Apply security best practices in all code development, especially for services handling sensitive data or external communication (e.g., `trading-gateway-service`, `portfolio-management-service`).