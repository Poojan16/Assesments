// ProtectedComponent.js
import React from 'react';
import { useSelector } from 'react-redux';
const ProtectedComponent = ({ requiredRoles, children }) => {
  const userRoles = useSelector((state) => state.auth.roles);
  const hasAccess = requiredRoles.some((role) => userRoles.includes(role));
  return hasAccess ? children : null;
};
export default ProtectedComponent;