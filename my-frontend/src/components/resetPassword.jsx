import React, { useEffect, useState } from 'react';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import forgotPassword from '../assets/forgotPassword.png';
import OtpInput from "./otp";
import { useLocation, useNavigate, useParams } from 'react-router-dom';



const setSessionItemWithExpiry = (key, value, minutes) => {
  const now = new Date();
  const expiryTime = now.getTime() + minutes * 60 * 1000; // Calculate expiry time in milliseconds
  const item = {
    value: value,
    expiry: expiryTime,
  };
  sessionStorage.setItem(key, JSON.stringify(item));
};

const ForgotPasswordForm = () => {
  const {email, token} = useParams();
  const navigate = useNavigate()
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [Data, setData] = useState("")
  const isDefined = token !== undefined && token !== undefined;
  console.log(isDefined)
  const [successMessage, setSuccessMessage] = useState(null);
  const location = useLocation();
  const prefilledEmail = location.state?.email || "";

  console.log(typeof email, typeof token)
  console.log(isDefined)

  useEffect(() => {
    if(token){
      const reset = document.getElementById('reset');
      const sendOtp = document.getElementById('sendOtp');
      sendOtp.classList.remove('hidden')
      reset.classList.add('hidden')
      
    }
  })

  const backend_url = process.env.REACT_APP_BACKEND_URL

  // Define the validation schema using Yup
  const validationSchema = Yup.object({
    email: Yup.string()
      .email('Invalid email address')
      .required('Email address is required'),
  });

  useEffect(() => {
      document.title = 'Reset Password'; // Set the desired title here
    }, []);

  const formik = useFormik({
    initialValues: {
      email: prefilledEmail || '',
    },
    validationSchema: validationSchema,
    onSubmit: async (values, { setSubmitting, resetForm },e) => {
      try {
        // Simulate API call to send reset email
        console.log('Sending password reset email for:', values.email);
        // Replace with actual API call (e.g., axios.post('/api/forgot-password', values))
        const response =  await fetch(`${backend_url}/users/link?email=${values.email}`, {
          method: 'GET',
        })
        const data = await response.json();
        if(!response.ok){
          setError(data.message);
          setMessage('');
        }
        if(data?.data?.token){
          console.log(data?.data?.token);
          const resetPassword = await fetch(`${backend_url}/users/resetPassword?token=${data?.data?.token}`, {
            method: 'GET',
          })
          const resetData = await resetPassword.json();
          if(!resetPassword.ok){
            setError(data.message);
            setMessage('');
          }
          if(resetData?.status_code === 200){
            setMessage("A link is sent to your email address for reset password");
            setTimeout(() => {
              navigate('/')
            }, 5000);
          }
        }
      } catch (err) {
        setMessage('');
      } finally {
        setSubmitting(false);
      }
    },
  });

  

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="flex px-8 py-6 mt-4 p-32 text-left bg-white shadow-lg rounded-lg">
        <div>
          <img src={forgotPassword} alt="" srcset="" />
        </div>
        <div className='w-1/2 py-[150px]'>
          <h3 className="text-2xl font-bold text-left">Forgot Password?</h3>
          <p className="mt-2 text-left text-gray-600">Enter your email address to receive a reset link.</p>
          <form onSubmit={formik.handleSubmit}>
            <div className="mt-4">
              {successMessage && <p className="text-red-500 mb-4">{successMessage} try again</p>}
              <div>
                <label htmlFor="email" className="block">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  placeholder="Email Address"
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  value={(email) ? email : formik.values.email}
                  className={`w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 ${
                    formik.touched.email && formik.errors.email
                      ? 'border-red-500 focus:ring-red-500'
                      : 'border-gray-300 focus:ring-blue-500'
                  }`}
                />
                {formik.touched.email && formik.errors.email && (
                  <p className="text-sm text-red-500 mt-1">{formik.errors.email}</p>
                )}
              </div>

              <div id='reset' hidden={!isDefined} className="flex items-baseline justify-between">
                <button
                  type="submit"
                  disabled={formik.isSubmitting}
                  className="w-full px-6 py-2 mt-4 text-white bg-blue-600 rounded-md hover:bg-blue-900 disabled:opacity-50"
                >
                  {formik.isSubmitting ? 'Sending...' : 'Send Reset Link'}
                </button>
              </div>
              {message && <p className="text-sm text-green-500 mt-4">{message}</p>}
              {error && <p className="text-sm text-red-500 mt-4">{error}</p>}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ForgotPasswordForm;
