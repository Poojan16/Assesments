// src/pages/UnauthorizedPage.jsx
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const UnauthorizedPage = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Optional: automatically redirect after a few seconds if you want
    // the user to see the message first.
    const timer = setTimeout(() => {
    //   navigate('/login');
    }, 3000); // Redirect after 3 seconds

    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="text-center p-8 bg-white shadow-lg rounded-lg">
        <h1 className="text-9xl font-extrabold text-indigo-600 mb-4">401</h1>
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Unauthorized Access</h2>
        <p className="text-gray-600 mb-6">
          You do not have permission to view this page. Redirecting to the login page...
        </p>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition duration-300"
        >
          Go to Login
        </button>
      </div>
    </div>
  );
};

export default UnauthorizedPage;
