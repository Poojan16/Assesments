import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm, Watch } from 'react-hook-form';
import { useDispatch, useSelector } from 'react-redux';
import { loginSuccess } from '../authSlice';
import Loader from './loader';
import CarRotationActivity from './botValidation';
import { object } from 'yup';
import {loadState}  from '../authSlice';


const LoginPage = () => {
  // const [email, setEmail] = useState('');
  // const [password, setPassword] = useState('');
  const [serverError, setServerError] = useState("");
  const navigate =  useNavigate()
  const dispatch = useDispatch();
  const [successMessage, setSuccessMessage] = useState(null);
  const [loading, setLoading] = useState(true)
  const [Data, setData] = useState({});
  console.log(Data)
  const { 
      register, 
      handleSubmit, 
      formState: { errors },
      clearErrors,
      watch, 
    } = useForm();

    const email = watch('email');

    const auths = localStorage.getItem('authState');

    const backend_url = process.env.REACT_APP_BACKEND_URL;

  const onSubmit = async (data) => {
    setLoading(true)
        try {
          const url = `${backend_url}/users/login`;
          const method = "POST";
        
          const response = await fetch(url, {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
          });
        
          console.log(response)

          if (!response.ok) {
            const message = "Login failed: " + response.statusText;
            setServerError(message);
            // timeout
            const timer = setTimeout(() => {
              setServerError(null);
              window.location.reload();
            }, 5000); // 5 seconds
            return;
          }
        
          data = await response.json();
        
          const userRole = data?.data || {};
          console.log(userRole);
          dispatch(loginSuccess({
            data: userRole,
            token: userRole.token,
            mark: data.mark,
            expired: userRole.expiresAt,
          }));

          console.log(data)

          if(data?.statusCode === 200){
            handleFormData(data)
          }else{
            setServerError(data?.detail);
          }
        } catch (error) {
          console.error("Login failed:", error);
        }
        finally{
          setLoading(false)
        }
      
    
  }; 
 
  const handleForm = (data) => {
    console.log(data)
    if(data){
      if (Data?.mark === "A") {
        setSuccessMessage(Data?.message)
        const timer = setTimeout(() => {
          setSuccessMessage(null);
          navigate("/admin");
        }, 5000); // 10 seconds

        return () => clearTimeout(timer);
      } else if (Data?.mark === "HT") {
        setSuccessMessage(Data?.message)
        const timer = setTimeout(() => {
          setSuccessMessage(null);
          navigate("/head-teacher");
        }, 5000); // 10 seconds

        return () => clearTimeout(timer);
      }else if (Data?.mark === "ST") {
        setSuccessMessage(Data?.message)
        const timer = setTimeout(() => {
          setSuccessMessage(null);
          navigate("/class-teacher");
        }, 5000); // 10 seconds

        return () => clearTimeout(timer);
      }
      Data = {} 
    }else{
      setServerError("Login failed: Bot validation failed.");
      const timer = setTimeout(() => {
        setServerError(null);
        window.location.reload();
      }, 3000);
    }
  }

  const handleFormData = (data) => {
    const form = document.getElementById('form')
    const bot = document.getElementById('bot')
    console.log(data)
    setData(data)
    if(bot && form){
      form.classList.add('hidden')
      bot.classList.remove('hidden')
    }
  }

  useEffect(()=>{
    if(Data?.statusCode === 200){
      handleFormData(Data)
    }
  },[Data])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className='flex flex items-center justify-center bg-white rounded-lg shadow-md p-5'>
        <div className='flex flex-col items-center bg-white p-8 py-[60px] w-full max-w-md'>
          <img src="https://rawafedschool.com/wp-content/uploads/2024/06/AnyConv.com__the-boy-was-very-happy.webp" alt="" srcset="" />
          <p className='text-2xl italic mt-5'>Student Grading System</p>
        </div>
        <div>
          {serverError && <p className="text-red-500 mb-4">{serverError}</p>}
          {successMessage && <p className="text-green-500 mb-4">{successMessage}</p>}
          <div id='form' className="bg-white p-8 w-full  max-w-md">
            <form onSubmit={handleSubmit(onSubmit)}>
              <div className="mb-4">
                <h2 className='text-2xl font-bold mb-6 text-center'>Login</h2>
                <label className="block text-gray-700 mb-2">Email</label>
                <input
                  id="email"
                  type="email"
                  className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    errors.email ? 'border-red-500' : 'border-gray-300'
                  }`}
                  onChange={() => {
                    clearErrors('email')
                    setServerError("")
                  }}
                  {...register('email', {
                    required: 'Email is required',
                    pattern: {
                      value:  /^[a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
                      message: 'Invalid email address',
                    },
                  })}
                />
                {errors.email && <p className="text-red-500 text-sm mt-1">{errors.email.message}</p>}
              </div>

              <div className="mb-4">
                <label className="block text-gray-700 mb-2">Password</label>
                <input
                    id="password"
                    type="password"
                    className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      errors.password ? 'border-red-500' : 'border-gray-300'
                    }`}
                    onChange={() => {
                      clearErrors('password')
                      setServerError("")
                    }}
                    {...register('password', {
                      required: "Password is required",
                      minLength: {
                        value: 8,
                        message: 'Password must be at least 8 characters long',
                      },
                      pattern: {
                        value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]).{8,}$/,
                        message: 'Password must contain one uppercase letter, one lowercase letter, one number, and one special character',
                      },
                    })}
                  />
                {errors.password && <p className="text-red-500 text-sm mt-1">{errors.password.message}</p>}
              </div>
              <div className="text-left mb-5">
                <a href="" onClick={() => navigate('/forgot-password', {
                  state: { email: email }
                })} className='text-blue-500 hover:underline'>Forgot Password?</a>
              </div>

              <button
                type="submit"
                className='login w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors'
              >
                Login
              </button>
            </form>

            {/* <p className="mt-4 text-center text-gray-600">
              Don't have an account?{' '}
              <Link to="/signup" className="text-blue-500 hover:underline">
                Sign Up
              </Link>
            </p> */}
          </div>
          <div id='bot' className=" max-w-3xl bot mt-4 mb-4 hidden">
            <CarRotationActivity onSubmitSuccess={handleForm}/>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;


