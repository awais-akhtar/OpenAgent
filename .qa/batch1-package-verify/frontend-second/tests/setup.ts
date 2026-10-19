import '@testing-library/jest-dom/vitest';


// JSDOM does not implement scrollTo. Motion calls it while measuring animated
// layouts, so provide a harmless test-environment stub instead of emitting a
// misleading not-implemented error during otherwise passing component tests.
window.scrollTo = () => undefined;

// Existing component/API tests exercise verified backend mode. Demo-specific tests
// can opt in by setting the isolated demo key in their own setup.
localStorage.setItem('oa-second-data-mode', 'live');
