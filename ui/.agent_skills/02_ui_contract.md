# 02 UI Contract

Use this file for page creation, page refactor, component split, UI consistency work, design system rollout, dashboard development, form flows, modal flows, table pages, responsive adaptation, or interaction cleanup.

## 1. When to use

Claude must read this file before making UI-facing changes in any of the following situations:

- Create a new page
- Refactor an existing page
- Split an oversized component
- Add or modify forms, modals, tables, dashboards, tabs, drawers, or detail panels
- Implement responsive layout changes
- Apply theme or design token unification
- Fix interaction consistency issues
- Standardize shared UI patterns across multiple pages

## 2. Inputs

Treat the following as development inputs, not algorithm inputs:

- Design source (Figma, screenshots, reference pages, issue description)
- Existing page file paths
- Component directories and shared UI library paths
- API interface definitions / request-response schemas
- Route definitions and navigation structure
- State management approach (local state / context / redux / zustand / query cache / etc.)
- Styling approach (CSS Modules / Tailwind / SCSS / CSS-in-JS / Ant Design token / custom theme system)
- Existing breakpoint rules and platform targets
- Accessibility requirements if specified by project or product

## 3. Outputs

Each UI task must produce a complete engineering deliverable set. Depending on scope, output should include:

- New page files or updated page files
- Component split results or extracted shared components
- Style files / token usage / layout adjustments
- Interaction logic updates
- Unit tests and/or interaction tests
- Documentation updates when patterns or conventions change

Do not perform visual-only edits without structural cleanup when the page is already unmaintainable.
Do not perform structural refactor without updating tests for critical flows.
Do not leave the task half-finished with placeholder handlers or incomplete UI states.

## 4. Hard Constraints

- Do not modify unauthorized directories.
- Do not break existing API field contracts without explicit approval.
- Do not introduce new global dependencies unless explicitly approved.
- Do not place business requests directly inside pure presentational components.
- Do not reimplement capabilities that already exist in shared components.
- Do not leave TODOs, placeholder buttons, dead clicks, or empty event handlers.
- Do not break existing desktop/mobile breakpoint rules.
- Do not mix temporary debug UI into production-facing components.
- Do not couple layout decisions to API response shape when a view model layer is needed.
- Do not hide loading / empty / error states behind implicit assumptions.

## 5. UI / Interaction Contract

### 5.1 Structure layering
- Separate page container logic from presentational components.
- Keep data fetching, transformation, and orchestration in container/page-level modules or hooks.
- Keep reusable visual blocks dumb where possible: props in, UI out.
- Extract repeated layout or action areas into shared components when reuse is real, not speculative.

### 5.2 State model
All user-facing pages must explicitly define these states where relevant:
- idle
- loading
- success
- empty
- error

Do not rely on `null` or `[]` alone to implicitly represent all states.

### 5.3 Form contract
- Define initial values explicitly.
- Distinguish create mode vs edit mode.
- Keep validation rules centralized and readable.
- Show field-level validation near the field.
- Show submit-level failure near the action area or top-level feedback zone.
- Reset behavior must be deterministic and testable.

### 5.4 Modal / drawer contract
- Opening source must be traceable.
- Closing conditions must be explicit.
- Confirm/cancel behavior must be consistent.
- Async submit inside modal must guard against duplicate submission.
- Modal state must not be fragmented across too many unrelated local states.

### 5.5 Table / list contract
- Columns and row actions must be explicit and stable.
- Empty state must be visible and meaningful.
- Loading state must not cause layout jump if avoidable.
- Dangerous actions must have confirmation or protective UX.
- Batch actions and row actions must follow consistent affordance.

### 5.6 Visual hierarchy
- Heading levels must reflect page structure.
- Spacing, typography, and grouping must use project tokens/rules.
- Avoid ad hoc pixel values when tokens or shared spacing scale exist.
- Primary, secondary, warning, and danger actions must be visually distinguishable and semantically correct.

### 5.7 Accessibility baseline
- Buttons, links, inputs, dialogs must be keyboard reachable.
- Form inputs must have accessible labels.
- Error messaging must be discoverable and associated with the relevant field/section.
- Focus order must remain logical after modal open/close and dynamic content updates.
- Avoid clickable non-semantic elements unless accessibility semantics are added.

## 6. Validation Rules

Before completion, validate all applicable items:

- Page renders successfully
- Empty data produces a real empty state
- Error state produces fallback UI
- Loading behavior is explicit and non-confusing
- No accidental duplicate requests are triggered
- No uncleaned side effects remain
- No visual overflow or broken layout appears at supported breakpoints
- No dead buttons / dead links / dead actions remain
- Lint passes
- Typecheck passes
- Tests pass

## 7. Failure Modes

Common UI failure modes to actively prevent:

- Single oversized file mixing layout, requests, transformations, and handlers
- State fragmentation causing modal or panel confusion
- Duplicate requests causing flicker or stale UI
- Deep props drilling caused by weak component boundaries
- Local style overrides breaking shared/global behavior
- Form reset not clearing derived state
- Async response overwriting newer state
- Repeated business logic copied across pages
- Shared component API becoming too broad and inconsistent
- Hidden coupling between route params, local state, and fetch timing

## 8. Required Tests

Add or update the following tests as applicable:

- Page render test
- Critical interaction test
- Form submission flow test
- Error state test
- Empty state test
- Responsive layout snapshot or breakpoint-focused test
- Regression test covering the modified user journey

After each UI refactor, Claude must also output a short root-cause review containing:

- The core maintainability problem in the original page
- Why the previous design was hard to maintain
- Which refactor pattern was applied this time
- Which anti-patterns must not appear again
- Which rules should be promoted into long-term project conventions
