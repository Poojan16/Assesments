import { createSlice } from '@reduxjs/toolkit';

const backend_url = process.env.REACT_APP_BACKEND_URL;

const loadState = () => {
  try {
    const serializedState = localStorage.getItem('authState');
    if (!serializedState) {
      return undefined;
    }
    
    const state = JSON.parse(serializedState);
    
    if (!state.token) {
      localStorage.removeItem('authState');
      return undefined;
    }
    
    return state;
  } catch (err) {
    localStorage.removeItem('authState');
    return undefined;
  }
};

export const checkSessionStatus = async (token) => {
  if (!token) return { expired: true, valid: false };
  
  try {
    const response = await fetch(`${backend_url}/sessions?sessionId=` + token);
    
    if (!response.ok) {
      return { expired: true, valid: false };
    }
    
    const responseData = await response.json();
    const sessions = responseData?.data || [];
    
    return {
        expired: sessions.expired,
        valid: sessions.valid,
        expiresAt: sessions.expiresAt
    };
  } catch (error) {
    console.error('Session check error:', error);
    return { expired: true, valid: false };
  }
};

const initialState = loadState() || {
  isAuthenticated: false,
  user: null,
  token: null,
  roles: [],
  mark: null,
  sessionExpired: false,
  sessionChecked: false,
  isLoading: true, // Add loading state
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginSuccess: (state, action) => {
      const payload = action.payload;
      const userData = payload.data || payload;
      
      state.isAuthenticated = true;
      state.user = userData;
      state.token = userData.token || payload.token;
      state.mark = payload.mark || null;
      state.roles = payload.roles || [];
      state.sessionExpired = false;
      state.sessionChecked = true;
      state.isLoading = false;
      
      localStorage.setItem('authState', JSON.stringify({
        isAuthenticated: true,
        user: state.user,
        token: state.token,
        roles: state.roles,
        mark: state.mark,
        sessionExpired: false,
        sessionChecked: true,
      }));
    },
    
    logout: (state) => {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
      state.roles = [];
      state.mark = null;
      state.sessionExpired = true;
      state.sessionChecked = true;
      state.isLoading = false;
      
      localStorage.removeItem('authState');
    },
    
    setSessionStatus: (state, action) => {
      const { expired, checked = true } = action.payload;
      state.sessionExpired = expired;
      state.sessionChecked = checked;
      state.isLoading = false;
      
      if (expired) {
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
        state.roles = [];
        state.mark = null;
        localStorage.removeItem('authState');
      } else {
        const currentState = localStorage.getItem('authState');
        if (currentState) {
          const parsedState = JSON.parse(currentState);
          parsedState.sessionExpired = expired;
          parsedState.sessionChecked = checked;
          localStorage.setItem('authState', JSON.stringify(parsedState));
        }
      }
    },
    
    updateSession: (state, action) => {
      if (action.payload.user) {
        state.user = { ...state.user, ...action.payload.user };
      }
      if (action.payload.mark !== undefined) {
        state.mark = action.payload.mark;
      }
      if (action.payload.roles) {
        state.roles = action.payload.roles;
      }
      
      const currentState = JSON.parse(localStorage.getItem('authState') || '{}');
      localStorage.setItem('authState', JSON.stringify({
        ...currentState,
        user: state.user,
        mark: state.mark,
        roles: state.roles,
      }));
    },
    
    initializeAuth: (state, action) => {
      if (action.payload) {
        return {
          ...state,
          ...action.payload,
          sessionChecked: true,
          isLoading: false,
        };
      }
    },
    
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    
    restoreAuthState: (state, action) => {
      // Simply restore from localStorage without validation
      const savedState = loadState();
      if (savedState) {
        return { 
          ...state, 
          ...savedState,
          isLoading: false,
          // Keep sessionChecked as false initially - will be validated later
          sessionChecked: false,
        };
      }
    },
  },
});

// Async action to validate session on app load
export const validateSessionOnLoad = () => async (dispatch) => {
  dispatch(setLoading(true));
  
  const savedState = loadState();
  
  if (!savedState || !savedState.token) {
    dispatch(setLoading(false));
    return;
  }
  console.log('Saved state:',savedState);
  
  // Immediately restore state from localStorage (for fast UI)
  dispatch(restoreAuthState());
  
  // Then validate with backend in background
  try {
    const sessionStatus = await checkSessionStatus(savedState.token);
    
    console.log('Session status:', sessionStatus);
    
    if (sessionStatus.valid && !sessionStatus.expired) {
      console.log('Session is valid');
      // Session is valid, update state
      dispatch(setSessionStatus({ 
        expired: false, 
        checked: true 
      }));
    } else {
      console.log('Session is invalid');
      // Session is invalid, clear everything
      dispatch(logout());
    }
  } catch (error) {
    console.error('Session validation failed:', error);
    // On network error, keep user logged in (offline support)
    dispatch(setSessionStatus({ 
      expired: false, 
      checked: true 
    }));
  } finally {
    dispatch(setLoading(false));
  }
};

export const checkSessionExpiry = () => async (dispatch, getState) => {
  const state = getState().auth;
  
  if (!state.token || !state.isAuthenticated) {
    return;
  }
  
  try {
    const sessionStatus = await checkSessionStatus(state.token);
    
    dispatch(setSessionStatus({ 
      expired: !sessionStatus.valid || sessionStatus.expired, 
      checked: true 
    }));
    
    if (!sessionStatus.valid || sessionStatus.expired) {
      if (typeof window !== 'undefined') {
        // Only redirect if we're not already on login page
        if (!window.location.pathname.includes('/')) {
          window.location.href = '/';
        }
      }
    }
  } catch (error) {
    console.error('Session check failed:', error);
    // Don't logout on network error
  }
};

export const { 
  loginSuccess, 
  logout,
  setSessionStatus,
  updateSession,
  initializeAuth,
  setLoading,
  restoreAuthState,
} = authSlice.actions;

// Updated middleware - only redirect on specific protected actions
export const sessionCheckMiddleware = (store) => (next) => (action) => {
  const state = store.getState().auth;
  
  // List of actions that should trigger redirect when session is expired
  const protectedActions = [
    // Add your protected action types here
    'api/callProtectedEndpoint',
    'dashboard/fetchData',
    // etc.
  ];
  
  // Only redirect on protected actions when session is expired
  if (state.sessionExpired && state.sessionChecked) {
    if (protectedActions.includes(action.type)) {
      console.log('Session expired, redirecting to login');
      if (typeof window !== 'undefined' && !window.location.pathname.includes('/')) {
        window.location.href = '/';
      }
      return;
    }
  }
  
  return next(action);
};

let sessionCheckInterval = null;

export const setupSessionMonitoring = (store) => {
  if (sessionCheckInterval) {
    clearInterval(sessionCheckInterval);
  }
  
  // Don't check immediately - wait a bit
  sessionCheckInterval = setInterval(() => {
    store.dispatch(checkSessionExpiry());
  }, 5 * 60 * 1000); // 5 minutes
  
  return () => {
    if (sessionCheckInterval) {
      clearInterval(sessionCheckInterval);
      sessionCheckInterval = null;
    }
  };
};

// Updated selectors
export const selectIsAuthenticated = (state) => 
  state.auth.isAuthenticated && 
  !state.auth.sessionExpired;

export const selectIsAuthLoading = (state) => state.auth.isLoading;
export const selectAuthToken = (state) => state.auth.token;
export const selectAuthUser = (state) => state.auth.user;
export const selectSessionExpired = (state) => state.auth.sessionExpired;
export const selectSessionChecked = (state) => state.auth.sessionChecked;
export const selectAuthMark = (state) => state.auth.mark;

export const selectAuthState = (state) => ({
  isAuthenticated: selectIsAuthenticated(state),
  user: state.auth.user,
  token: state.auth.token,
  mark: state.auth.mark,
  sessionExpired: state.auth.sessionExpired,
  sessionChecked: state.auth.sessionChecked,
  isLoading: state.auth.isLoading,
});

export default authSlice;
