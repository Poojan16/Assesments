    // // features/auth/authSlice.js
    // import { createSlice } from '@reduxjs/toolkit';

    // const initialState = {
    //     isAuthenticated: false,
    //     user: null, // { id, username, role, permissions: [] }
    //     token: null,
    // };

    // const authSlice = createSlice({
    //     name: 'auth',
    //     initialState,
    //     reducers: {
    //         loginSuccess: (state, action) => {
    //             state.isAuthenticated = true;
    //             state.user = action.payload;
    //         },
    //         logout: (state) => {
    //             state.isAuthenticated = false;
    //             state.user = null;
    //             state.token = null;
    //         },
    //         // You might also have a reducer to update user permissions if they change during a session
    //     },
    // });

    // export const { loginSuccess, logout } = authSlice.actions;
    // export default authSlice;


// authSlice.js
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