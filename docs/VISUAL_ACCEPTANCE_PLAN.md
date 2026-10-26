# G18: Desktop Visual Acceptance Plan

## Current State

**Canonical design SHA-256**: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`

- Original 23 screens: Historical visual pairs exist
- New 4 screens: Require separate acceptance (Image Studio, Model Cookbook, Ultra Model, Personal OS)
- Coverage: Both 1440×900 and 1280×800 viewports, light and dark themes

## Viewport Matrix

| Viewport | Width | Height | Usage | Acceptance |
|----------|-------|--------|-------|-----------|
| Desktop Primary | 1440 | 900 | Main development | Yes |
| Desktop Alt | 1280 | 800 | Laptop/older displays | Yes |
| Tablet | 768 | 1024 | For reference only | Future |
| Mobile | 390 | 844 | Explicit when requested | Future |

## Screen Acceptance Status

### Original 23 Screens (Historical)
- Home, Chat, Agents, Research, Files, Automations, Calendar, Models
- Settings, CodeQ, Knowledge, Voice, Assistant, Communications, Contacts
- Security, Developer, Science, Personal Context, Vault, Scraper, Fine-tuning
- Status: Visual pairs documented; design SHA matches

### New 4 Screens (Need Acceptance)
- Image Studio: New gallery view
- Model Cookbook: Model serving interface
- Ultra Model: Model runtime controls
- Personal OS: Context export and presets

## Acceptance Process

### Phase 1: Capture Current State
```bash
# For each new screen, capture:
# 1. Idle state (1440x900 light)
# 2. Idle state (1440x900 dark)
# 3. Idle state (1280x800 light)
# 4. Idle state (1280x800 dark)
# 5. Dialog states
# 6. Error states
# 7. Empty states
```

### Phase 2: Compare to Approved Design
- Verify component alignment
- Check color contrast (WCAG AA)
- Confirm typography matches
- Validate spacing and layout

### Phase 3: Mark Intentional Divergences
- Dialog layering
- Focus indicators
- Disabled state styling
- Loading animations

### Phase 4: Approve and Archive
- Generate golden master images
- Document approved states
- Create regression test baseline
- Archive with git

## New Screens Checklist

### Image Studio
- [ ] Gallery grid layout
- [ ] Album selector
- [ ] Image detail view
- [ ] Upscale/rotate dialogs
- [ ] Undo history list
- [ ] Empty gallery state
- [ ] Loading state

### Model Cookbook
- [ ] Host selector
- [ ] Serve plan creation
- [ ] Service logs panel
- [ ] Stop/restart buttons
- [ ] Process status indicator
- [ ] Error messages

### Ultra Model
- [ ] Manifest registry
- [ ] Launch controls
- [ ] Session list
- [ ] Stop/calibrate buttons
- [ ] Model status indicator
- [ ] Demo isolation verification

### Personal OS
- [ ] Status/export section
- [ ] Preset creator
- [ ] Preset list
- [ ] Gallery browser
- [ ] Draft signatures
- [ ] Empty states

## Measurement Criteria

### Layout Accuracy
- Component alignment (±2px)
- Spacing consistency (±1px)
- Text overflow handled
- Scrollable regions work

### Color Consistency
- WCAG AA contrast verified
- Color tokens match design
- Dark mode inversion correct
- Hover states visible

### Typography
- Font sizes exact
- Line heights correct
- Font weights match
- Text truncation handled

### States
- Hover effects present
- Focus visible
- Disabled clearly marked
- Loading animations smooth

## Regression Testing

After acceptance, verify with:
```bash
# Visual regression test
npm run test:visual -- --update

# Component storybook
npm run storybook

# Manual spot checks at each viewport
npm run dev
# Then test in browser at each size
```

## Known Approved Divergences

(Documented differences from initial design that are intentional)

- Context trace suppression in certain views (intentional)
- Loading spinner placement
- Error message formatting
- Empty state messaging

## Approval Sign-Off

Requires:
1. Visual inspection by designer
2. Cross-viewport verification
3. Theme verification (light/dark)
4. Accessibility check (contrast, focus)
5. Document in APPROVAL.md

## Timeline

- Current: Capture baseline for new 4 screens
- Week 2: Designer review and approval
- Week 3: Archive golden masters
- Week 4: Establish regression baseline
