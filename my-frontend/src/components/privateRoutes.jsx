    // components/ProtectedRoute.jsx
    import React, { useEffect, useState } from 'react';
    import { useSelector } from 'react-redux';
    import { Navigate, Outlet } from 'react-router-dom';
    import { logout } from '../authSlice';

    const ProtectedRoute = ({ allowedRoles = [] }) => {
        const { isAuthenticated, user } = useSelector((state) => state.auth);

        if (!isAuthenticated) {
            return <Navigate to="/" replace />;
        }

        if (allowedRoles.length > 0 && !allowedRoles.includes(user?.role)) {
            return <Navigate to="/unauthorized" replace />;

        }

        return <Outlet />;
    };

    export default ProtectedRoute;