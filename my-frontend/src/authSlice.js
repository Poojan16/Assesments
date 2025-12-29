import { createSlice } from '@reduxjs/toolkit';

// Load initial state from localStorage
const loadState = () => {
  try {
    const serializedState = localStorage.getItem('authState');
    if (serializedState === null) {
      return undefined;
    }
    return JSON.parse(serializedState);
  } catch (err) {
    return undefined;
  }
};

const initialState = loadState() || {
  isAuthenticated: false,
  user: null,
  token: null,
  roles: [],
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginSuccess: (state, action) => {
      state.isAuthenticated = true;
      console.log(action.payload)
      state.user = action.payload;
      
      // Save to localStorage
      localStorage.setItem('authState', JSON.stringify(state));
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
      
      // Clear localStorage
      localStorage.removeItem('authState');
    },
    initializeAuth: (state) => {
      const savedState = loadState();
      if (savedState) {
        return { ...state, ...savedState };
      }
    },
  },
});

export const { loginSuccess, logout, initializeAuth } = authSlice.actions;
export default authSlice;