import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import Loader from "./loader";
import  signUpImg  from "../assets/4474436.jpg";
import OtpInput from "./otp";
import PasswordStrengthIndicator from "./password";

const setSessionItemWithExpiry = (key, value, minutes) => {
  const now = new Date();
  const expiryTime = now.getTime() + minutes * 60 * 1000; // Calculate expiry time in milliseconds
  const item = {
    value: value,
    expiry: expiryTime,
  };
  sessionStorage.setItem(key, JSON.stringify(item));
};

const SignupPage = () => {
  const navigate = useNavigate();
  const [roles, setRoles] = useState([]);
  const [serverError, setServerError] = useState("");
  const [successMessage, setSuccessMessage] = useState(null);
  const [loading,setLoading] = useState(true)
  const [Data, setData] = useState("")

  useEffect(() => {
      document.title = 'Sign Up'; // Set the desired title here
    }, []);
  

  // Fetch roles from API
  useEffect(() => {
    const fetchRoles = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/roles/", {
          method: "GET"

        });
        const data = await response.json();
        console.log(data);
        setRoles(data); // expecting [{id:1, name:'Admin', mark:'admin'}, ...]
      } catch (error) {
        console.error("Failed to fetch roles", error);
      }
    };
    fetchRoles();
  }, []);

  const {
    register,
    handleSubmit,
    watch,
    clearErrors,
    formState: { errors },
  } = useForm({
    mode: "onBlur",
  });

  const isFormValid = Object.keys(errors).length === 0;
  const allFields = watch();
  const areFieldsFilled = Object.keys(allFields).every((field) => allFields[field] !== "" && allFields[field] !== null && allFields[field] !== undefined);
  const sendOtp = document.getElementById("sendOtp");
  console.log(sendOtp)

  if(isFormValid && areFieldsFilled && sendOtp){
    sendOtp.classList.remove("hidden")
  }

  const password = watch("password");

  useEffect(() => {
    if(password && password.length > 0){
      document.getElementById("password-strength-indicator").classList.remove("hidden")
    }
  }, [password]);

  const onSubmit = async (data,e) => {
    e.preventDefault();
    const now = new Date();
    const otp = JSON.parse(sessionStorage.getItem("SignupOtp"));
    if (otp && now.getTime() > otp.expiry) {
      sessionStorage.removeItem("SignupOtp"); // Remove expired item
      return setServerError("OTP expired. Please request a new one.");
    }
    console.log("OTP:", otp)
    console.log("Data:", Data)
    const correctOtp = otp.value === Data;
    if(correctOtp){
      try {
        const response = await fetch("http://127.0.0.1:8000/users/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
  
        if (!response.ok) {
          const errorData = await response.json();
          const message = Array.isArray(errorData.message)
            ? errorData.message.join(", ")
            : errorData.message || "Submission failed";
          setServerError(message);
          return;
        }
          setSuccessMessage("New user created.")
          const timer = setTimeout(() => {
            setSuccessMessage(null);
            navigate("/login");
          }, 2000); // 10 seconds
  
          return () => clearTimeout(timer); // Cleanup on unmount or message change
          
        
  
      } catch (error) {
        setServerError(error.message);
      }
    }else{
      setServerError("Invalid OTP")
      sendOtp.textContent = "Resend OTP";
    }
  };

  const handleSendOtp = (e) => {
    e.preventDefault();
    const url = "http://127.0.0.1:8000/users/otp?email=" + document.getElementById("email").value;
    const method = "GET";
    fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        const otpElement = document.getElementById("otp");
        otpElement.classList.remove('hidden')
        if(data.otp){
          setSessionItemWithExpiry("SignupOtp", JSON.stringify(data.otp),1);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };


  const handleComplete = (data) => {
    console.log(data)
    const submit = document.getElementById("submit");
    setData(data)

  };

  const handleCancel = () => navigate("/login");

  return (
    <div className="min-h-screen mx-auto p-4 flex items-center justify-center py-screen bg-gray-100">
      <div className="max-w-7xl flex bg-white p-3 rounded-lg shadow-md">
        <div className="w-full bg-white w-full mr-5">
          <img src={signUpImg} alt="" className="w-full rounded-lg shadow-md object-fit h-full w-full" srcset="" />
        </div>
        <div className="w-full max-w-lg">
          <h2 className="text-2xl font-bold mb-6 text-center">Sign Up</h2>

          {serverError && (
            <div className="mb-4 p-2 bg-red-100 text-red-700 border border-red-300 rounded">
              {serverError}
            </div>
          )}
          <p className='m-4 text-green-600'>{successMessage}</p>

          <form onSubmit={handleSubmit(onSubmit)} noValidate>
            {/* Name */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Name</label>
              <input
                type="text"
                {...register("userName", {
                  required: "Name is required",
                  minLength: {
                    value: 2,
                    message: "Name must be at least 2 characters",
                  },
                })}
                onChange={() => {
                  clearErrors('userName')
                }}
                className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.userName ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.userName && (
                <p className="text-red-500 text-sm mt-1">{errors.userName.message}</p>
              )}
            </div>

            {/* Email */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Email</label>
              <input
              id="email"
                type="email"
                {...register("userEmail", {
                  required: "Email is required",
                  pattern: {
                    value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
                    message: "Email is invalid",
                  },
                })}
                onChange={async (e) => {
                  clearErrors('userEmail');
                  const response = await fetch(`http://localhost:3000/v1/users/${e.target.value}`, {
                    method: "GET",
                    headers: { "Content-Type": "application/json" },
                  });
                  const data = await response.json();
                  console.log(data?.role)
                  if (data?.role) {
                    setServerError("Email already exists");
                  }else {
                    console.log(data?.message)
                    setServerError("")
                  }
                }}
                className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.userEmail ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.userEmail && (
                <p className="text-red-500 text-sm mt-1">{errors.userEmail.message}</p>
              )}
            </div>

            {/* Password */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Password</label>
              <input
                type="password"
                onChange={() => {
                  clearErrors('password')
                }}
                {...register("password", {
                  required: "Password is required",
                  minLength: {
                    value: 8,
                    message: "Password must be at least 8 characters",
                  },
                  validate: {
                    hasUpper: (v) =>
                      /[A-Z]/.test(v) ||
                      "Password must contain at least one uppercase letter",
                    hasLower: (v) =>
                      /[a-z]/.test(v) ||
                      "Password must contain at least one lowercase letter",
                    hasNumber: (v) =>
                      /[0-9]/.test(v) ||
                      "Password must contain at least one number",
                    hasSpecial: (v) =>
                      /[!@#$%^&*(),.?":{}|<>]/.test(v) ||
                      "Password must contain at least one special character",
                  },
                })}
                className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.password ? "border-red-500" : "border-gray-300"
                }`}
              />
              <div id="password-strength-indicator" className="hidden">
                <PasswordStrengthIndicator password={password} />
              </div>
              {errors.password && (
                <p className="text-red-500 text-sm mt-1">
                  {errors.password.message}
                </p>
              )}
            </div>

            {/* Confirm Password */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Confirm Password</label>
              <input
                type="password"
                onChange={() => {
                  clearErrors('pconfirmPassword')
                }}
                {...register("confirmPassword", {
                  required: "Confirm password is required",
                  validate: (value) =>
                    value === password || "Passwords do not match",
                })}
                className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.confirmPassword ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.confirmPassword && (
                <p className="text-red-500 text-sm mt-1">
                  {errors.confirmPassword.message}
                </p>
              )}
            </div>

            {/* Role */}
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Role</label>
              <select
              onChange={() => {
                clearErrors('role')
              }}
                {...register("role", { required: "Role is required" })}
                className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.role ? "border-red-500" : "border-gray-300"
                }`}
              >
                <option value="">Select Role</option>
                {roles.map((role) => (
                  <option key={role.roleId} value={role.mark}>
                    {role.roleName}
                  </option>
                ))}
              </select>
              {errors.role && (
                <p className="text-red-500 text-sm mt-1">{errors.role.message}</p>
              )}
            </div>

            <div id="otp" className="mb-4 hidden">
              <label className="block text-gray-700 mb-2">OTP Verification</label>
              <OtpInput length={7} onComplete={handleComplete} />
            </div>

            <button
                onClick={(e) => {
                  handleSendOtp(e);
                }}
                id="sendOtp"
                className="bg-blue-500 text-white py-2 px-6 rounded-lg hover:bg-blue-600 transition-colors hidden"
              >
                Send OTP
              </button>

            {/* Buttons */}
            <div className="flex justify-between mt-6">
              <button
              id="submit"
                type="submit"
                className="bg-blue-500 text-white py-2 px-6 rounded-lg hover:bg-blue-600 transition-colors"
              >
                Save
              </button>
              <button
                type="button"
                onClick={handleCancel}
                className="bg-gray-300 text-gray-700 py-2 px-6 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
