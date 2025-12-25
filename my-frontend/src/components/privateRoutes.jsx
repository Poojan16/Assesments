    // components/ProtectedRoute.jsx
    import React from 'react';
    import { useSelector } from 'react-redux';
    import { Navigate, Outlet } from 'react-router-dom';

    const ProtectedRoute = ({ allowedRoles = [] }) => {
        const { isAuthenticated, user } = useSelector((state) => state.auth);
        console.log("User",user)

        if (!isAuthenticated) {
            return <Navigate to="/" replace />;
        }

        if (allowedRoles.length > 0 && !allowedRoles.includes(user?.role)) {
            return <Navigate to="/unauthorized" replace />;

        }

        return <Outlet />;
    };

    export default ProtectedRoute;