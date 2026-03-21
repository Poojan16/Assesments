// import { configureStore } from '@reduxjs/toolkit';
// import authSlice from './authSlice';
// export const store = configureStore({
//     reducer: {
//         auth: authSlice.reducer,
//     },
// });

import { configureStore } from '@reduxjs/toolkit';
import authReducer, { sessionCheckMiddleware, setupSessionMonitoring } from './authSlice';
import authSlice from './authSlice';

const store = configureStore({
  reducer: {
    auth: authSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(sessionCheckMiddleware),
});

// Setup session monitoring
setupSessionMonitoring(store);

export default store;