    // components/ProtectedRoute.jsx
    import React, { useEffect, useState } from 'react';
    import { useSelector } from 'react-redux';
    import { Navigate, Outlet } from 'react-router-dom';
    import { logout, selectIsAuthenticated, selectSessionChecked } from '../authSlice';

    const ProtectedRoute = ({ allowedRoles = [] }) => {
        const { user } = useSelector((state) => state.auth);
        const isAuthenticated = useSelector(selectIsAuthenticated);
        const sessionChecked = useSelector(selectSessionChecked);
        console.log(isAuthenticated,sessionChecked)

        if (!isAuthenticated) {
            return <Navigate to="/" replace />;
        }

        if (allowedRoles.length > 0 && !allowedRoles.includes(user?.role)) {
            return <Navigate to="/unauthorized" replace />;

        }

        return <Outlet />;
    };

    export default ProtectedRoute;