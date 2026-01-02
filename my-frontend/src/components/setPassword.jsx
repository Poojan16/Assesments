import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';

const NewPasswordForm = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errors, setErrors] = useState({});
  const [successMessage, setSuccessMessage] = useState('');
  const [serverError, setServerError] = useState("");
  const [isTokenValid, setIsTokenValid] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();
  const [tokenData, setTokenData] = useState(null);


  const token = (window.location.pathname).split('/').pop();
  console.log(token);

  const backend_url = process.env.REACT_APP_BACKEND_URL;

  useEffect(() => {
    document.title = 'Set Password';
    
    const validateToken = async () => {
      if (!token) {
        setServerError("Invalid or missing reset token");
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch(`${backend_url}/users/decodelink?link=${token}`);
        
        if (!response.ok) {
          throw new Error('Token validation failed');
        }
        
        const data = await response.json();
        console.log("Token validation response:", data);
        
        // Check if token is valid based on your API response
        // Adjust this condition based on your actual API response structure
        if (data.status_code === 200) {
          setTokenData(data);
          setIsTokenValid(true);
        } else {
          setServerError(data.detail || "Token is invalid or has expired or has already been used.");
        }
      } catch (error) {
        console.error("Token validation error:", error);
      } finally {
        setIsLoading(false);
      }
    };

    validateToken();
  }, [token]);

  console.log(tokenData);

  const validatePassword = (pwd) => {
    const newErrors = {};
    if (pwd.length < 8) {
      newErrors.length = 'Password must be at least 8 characters long.';
    }
    if (!/[A-Z]/.test(pwd)) {
      newErrors.uppercase = 'Password must contain at least one uppercase letter.';
    }
    if (!/[a-z]/.test(pwd)) {
      newErrors.lowercase = 'Password must contain at least one lowercase letter.';
    }
    if (!/[0-9]/.test(pwd)) {
      newErrors.number = 'Password must contain at least one number.';
    }
    if (!/[!@#$%^&*()]/.test(pwd)) {
      newErrors.specialChar = 'Password must contain at least one special character (!@#$%^&*()).';
    }
    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset errors
    setServerError("");
    
    // Validate password
    const passwordValidationErrors = validatePassword(password);
    const newErrors = { ...passwordValidationErrors };

    if (Object.keys(passwordValidationErrors).length > 0) {
      setErrors(newErrors);
      setSuccessMessage('');
      return;
    }

    if (password !== confirmPassword) {
      newErrors.confirm = 'Passwords do not match.';
      setErrors(newErrors);
      setSuccessMessage('');
      return;
    }

    // If we have a valid token, use it to reset password
    if (token && isTokenValid) {
      try {
        const formData = new FormData();
        formData.append('email', tokenData?.data?.email);
        formData.append('password', password);
        formData.append('token', token ? token : '');
        const response = await fetch(`${backend_url}/users/reset`, {
          method: 'PUT',
          body: formData
        });

        if (!response.ok) {
          const errorData = await response.json();
          const message = Array.isArray(errorData.message)
            ? errorData.message.join(', ')
            : errorData.message || 'Password reset failed';
          throw new Error(message);
        }

        const result = await response.json();
        console.log("Password reset successful:", result);
        
        setErrors({});
        setSuccessMessage('Password successfully updated!');
        setPassword('');
        setConfirmPassword('');
        
        // Redirect to login after success
        const timer = setTimeout(() => {
          navigate("/");
        }, 4000);
        
        return () => clearTimeout(timer);
        
      } catch (error) {
        console.error("Password reset error:", error);
        setServerError(error.message);
      }
    } else {
      setServerError("Invalid token. Cannot reset password.");
    }
  };

  // Show loading state while validating token
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded shadow-md w-full max-w-md text-center">
          <h2 className="text-2xl font-bold mb-4">Validating Reset Link...</h2>
          <p>Please wait while we verify your password reset link.</p>
        </div>
      </div>
    );
  }

  // Show error if token is invalid
  if (!isTokenValid) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded shadow-md w-full max-w-md">
          <h2 className="text-2xl font-bold mb-4 text-red-600">Invalid Reset Link</h2>
          <p className="mb-4">{serverError || "This password reset link is invalid or has expired."}</p>
          <button
            onClick={() => navigate('/forgot-password')}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Request New Reset Link
          </button>
        </div>
      </div>
    );
  }

  // Render the password reset form if token is valid
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white p-8 rounded shadow-md w-full max-w-md">
        <button 
          onClick={() => navigate('/forgot-password')}
          className="mb-4 flex items-center text-blue-600 hover:text-blue-800"
        >
          ← Back
        </button>
        <h2 className="text-2xl font-bold mb-6 text-center">Set New Password</h2>
        
        {serverError && (
          <div className="mb-4 p-3 bg-red-100 text-red-700 border border-red-300 rounded">
            {serverError}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700" htmlFor="password">
              New Password
            </label>
            <input
              type="password"
              id="password"
              className={`mt-1 block w-full px-3 py-2 border ${
                errors.length || errors.uppercase || errors.lowercase || errors.number || errors.specialChar
                  ? 'border-red-500'
                  : 'border-gray-300'
              } rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm`}
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                setErrors(validatePassword(e.target.value));
              }}
              required
              disabled={!isTokenValid}
            />
            {Object.values(errors).map((error, index) => (
              <p key={index} className="mt-1 text-sm text-red-500">
                {error}
              </p>
            ))}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700" htmlFor="confirmPassword">
              Confirm New Password
            </label>
            <input
              type="password"
              id="confirmPassword"
              className={`mt-1 block w-full px-3 py-2 border ${
                errors.confirm ? 'border-red-500' : 'border-gray-300'
              } rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm`}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              disabled={!isTokenValid}
            />
            {errors.confirm && <p className="mt-1 text-sm text-red-500">{errors.confirm}</p>}
          </div>

          {successMessage && (
            <div className="mt-4 p-3 bg-green-100 text-green-700 border border-green-300 rounded">
              {successMessage}
            </div>
          )}

          <button
            type="submit"
            disabled={!isTokenValid}
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              !isTokenValid 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
          >
            Set New Password
          </button>
        </form>
      </div>
    </div>
  );
};

export default NewPasswordForm;