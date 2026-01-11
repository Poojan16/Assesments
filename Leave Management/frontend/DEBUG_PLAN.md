# Frontend Debugging Plan

## Issues Identified

### 1. Tailwind CSS v4 Configuration Issues
- The project uses Tailwind CSS v4.1.18 which has a completely different setup than v3
- Current `postcss.config.js` uses `@tailwindcss/postcss` plugin
- The setup needs to be updated to work properly with Tailwind v4
- Need to update to use `@tailwindcss/vite` plugin for Vite

### 2. API Response Data Structure Issues
**Dashboard.jsx (Employee Stats):**
- Code makes separate API calls to get counts for each status
- `recentResponse.data?.total` is used but the backend returns `LeaveListResponse` with `total` directly
- Issue: `total` from `recentResponse.data?.total` gets `LeaveListResponse.total` which is the total for that specific query (limit 5), not all leaves

**Fix needed:** Need to call a separate endpoint or modify the approach to get accurate stats

### 3. Missing Manager Stats Endpoint for Employees
- Employees don't have a way to get their leave statistics
- The stats for employees are calculated by making multiple API calls which is inefficient

### 4. Missing Status Filter Counts
- `MyLeaves.jsx` has `statusFilterOptions` with `count: 0` hardcoded
- Should show actual counts for each status

## Files to Edit

1. `frontend/postcss.config.js` - Fix Tailwind CSS configuration for v4
2. `frontend/vite.config.js` - Add Tailwind CSS Vite plugin
3. `frontend/src/index.css` - Ensure proper Tailwind imports
4. `frontend/package.json` - Add @tailwindcss/vite dependency
5. `frontend/src/pages/Dashboard.jsx` - Fix stats calculation for employees
6. `frontend/src/pages/MyLeaves.jsx` - Fix status filter counts

## Implementation Steps

### Step 1: Fix Tailwind CSS Configuration
1. Install `@tailwindcss/vite` package
2. Update `vite.config.js` to use Tailwind CSS plugin
3. Update `postcss.config.js` to remove @tailwindcss/postcss
4. Update `index.css` to use proper v4 imports

### Step 2: Fix Dashboard Stats Calculation
For employees:
- Currently fetches recent leaves with limit=5
- Uses `response.data?.total` which is total for that query (5), not all leaves
- Fix: Get total from a separate call or use the response differently

### Step 3: Add Status Counts to MyLeaves
- Update the status filter options to show actual counts

## Testing After Fixes
1. Verify Tailwind classes render correctly (colors, spacing, etc.)
2. Check API data is received and displayed properly
3. Verify stats show correct numbers
4. Test all pages load and interact correctly

