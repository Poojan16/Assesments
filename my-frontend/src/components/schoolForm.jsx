import React, { useEffect, useState } from 'react';
import { useNavigate } from "react-router-dom";
import { useForm } from 'react-hook-form';
import { 
  UserPlus, Mail, Phone, MapPin, Calendar, BookOpen, Save, X, Upload,
  ChevronRight, ChevronLeft, Globe, Home, FileText 
} from 'lucide-react';
import axios from 'axios';
import Select from 'react-select';
import AsyncSelect from 'react-select/async';

export default function AddSchoolForm() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [roles, setRoles] = useState([]);
  const [currentStep, setCurrentStep] = useState(1);
  const [countries, setCountries] = useState([]);
  const [states, setStates] = useState([]);
  const [cities, setCities] = useState([]);
  const [pincodes, setPincodes] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [selectedState, setSelectedState] = useState(null);
  const [selectedCity, setSelectedCity] = useState(null);
  const [selectedPin, setSelectedPin] = useState(null);
  const [selectedRole, setSelectedRole] = useState(null);
  const [selectedQualification, setSelectedQualification] = useState(null);
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
    setValue,
    trigger,
    clearErrors
  } = useForm({
    defaultValues: {
        schoolName: '',
        schoolEmail: '',
        primaryContactNo: '',
        secondaryContactNo: '',
        additionalContactNo: '',
        cStatus: true,
        address: '',
        established_year: '',
      gender: '',
      country: '',
      state: '',
      city: '',
      pin: ''
    }
  });

  const [teachers, setTeachers] = useState([]);
  const [serverError, setServerError] = useState('');

  const steps = [
    { id: 1, name: 'Personal Info', icon: <UserPlus className="w-4 h-4" /> },
    { id: 2, name: 'Contact', icon: <Mail className="w-4 h-4" /> },
    { id: 3, name: 'Address', icon: <MapPin className="w-4 h-4" /> },
    { id: 4, name: 'Professional', icon: <BookOpen className="w-4 h-4" /> },
    { id: 5, name: 'Documents', icon: <FileText className="w-4 h-4" /> },
  ];

  // Load external CSS for react-select
  useEffect(() => {
    const link = document.createElement('link');
    link.href = 'https://cdn.jsdelivr.net/npm/react-select@5/dist/react-select.min.css';
    link.rel = 'stylesheet';
    document.head.appendChild(link);

    return () => {
      document.head.removeChild(link);
    };
  }, []);

  useEffect(() => {
    const handleRoles = async () => {
      try {
        const [rolesResponse, teacherResponse] = await Promise.all([
          fetch('http://127.0.0.1:8000/roles/'),
          fetch('http://127.0.0.1:8000/teachers/'),
        ]);

        const rolesData = await rolesResponse.json();
        const teacherData = await teacherResponse.json();

        setRoles(rolesData);
        setTeachers(teacherData);
      } catch (error) {
        console.error('Failed to fetch data', error);
      }
    };

    handleRoles();
    fetchCountries();
  }, []);

  // Fetch countries
  const fetchCountries = async () => {
    try {
      const response = await axios.get('https://countriesnow.space/api/v0.1/countries');
      const countriesData = response.data.data.map(country => ({
        value: country.country,
        label: country.country,
        iso2: country.iso2
      }));
      setCountries(countriesData);
    } catch (error) {
      console.error('Error fetching countries:', error);
    }
  };

  // Load states for selected country
  const loadStates = async (inputValue) => {
    if (!selectedCountry) return [];
    
    try {
      const response = await axios.post('https://countriesnow.space/api/v0.1/countries/states', {
        country: selectedCountry.value
      });
      
      const statesData = response.data.data.states || [];
      return statesData
        .filter(state => state.name.toLowerCase().includes(inputValue.toLowerCase()))
        .map(state => ({
          value: state.name,
          label: state.name
        }));
    } catch (error) {
      console.error('Error loading states:', error);
      return [];
    }
  };

  // Load cities for selected country and state
  const loadCities = async (inputValue) => {
    if (!selectedCountry || !selectedState) return [];
    
    try {
      const response = await axios.post('https://countriesnow.space/api/v0.1/countries/state/cities', {
        country: selectedCountry.value,
        state: selectedState.value
      });
      
      const citiesData = response.data.data || [];
      return citiesData
        .filter(city => city.toLowerCase().includes(inputValue.toLowerCase()))
        .map(city => ({
          value: city,
          label: city
        }));
    } catch (error) {
      console.error('Error loading cities:', error);
      return [];
    }
  };

  // Load pincodes for selected city
  const loadPincodes = async (inputValue) => {
    if (!selectedCity) return [];
    
    try {
      const response = await axios.get(`https://api.postalpincode.in/postoffice/${selectedCity.value}`);
      if (response.data[0].Status === 'Success') {
        const postOffices = response.data[0].PostOffice;
        const uniquePincodes = [...new Set(postOffices.map(office => office.Pincode))];
        
        return uniquePincodes
          .filter(pin => pin.includes(inputValue))
          .map(pin => ({
            value: pin,
            label: pin
          }));
      }
      return [];
    } catch (error) {
      console.error('Error loading pincodes:', error);
      return [];
    }
  };

  // Qualification options
  const qualificationOptions = [
    { value: 'B.Ed', label: 'B.Ed' },
    { value: 'M.Ed', label: 'M.Ed' },
    { value: 'B.A', label: 'B.A' },
    { value: 'M.A', label: 'M.A' },
    { value: 'B.Sc', label: 'B.Sc' },
    { value: 'M.Sc', label: 'M.Sc' },
    { value: 'Ph.D', label: 'Ph.D' },
    { value: 'M.Phil', label: 'M.Phil' },
    { value: 'B.Tech', label: 'B.Tech' },
    { value: 'M.Tech', label: 'M.Tech' },
  ];

  // Format roles for react-select
  const roleOptions = roles.map(role => ({
    value: role.roleId,
    label: role.roleName
  }));

  const handleEmail = async () => {
    try {
      const email = watch('teacherEmail');
      if(email) {
        const teacher = teachers.find((teacher) => teacher.teacherEmail === email);
        if (teacher) {
          setServerError('Email already exists');
        } else {
          setServerError('');
        }
      } else {
        setServerError('');
      }
    } catch (error) {
      console.error('Error checking email:', error);
    }
  };

  const onSubmit = async (data) => {
    setIsSubmitting(true);
    const form = document.querySelector('form');
    const formElements = form.elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = true;
    }
    const formData = new FormData();

    for (const key in data) {
      if (data[key] instanceof FileList) {
        if (data[key].length > 0) {
          formData.append(key, data[key][0]);
        }
      } else {
        formData.append(key, data[key]);
      }
    }
    
    const response = await fetch('http://127.0.0.1:8000/schools/', {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      setSubmitSuccess(true);
      setTimeout(() => {
        navigate('/head-teacher');
        handleReset();
      }, 3000);
    } else {
      console.error('Failed to add teacher');
    }
    
    setIsSubmitting(false);
  };

  const handleReset = () => {
    reset();
    setSubmitSuccess(false);
    setCurrentStep(1);
    setSelectedCountry(null);
    setSelectedState(null);
    setSelectedCity(null);
    setSelectedPin(null);
    setSelectedRole(null);
    setSelectedQualification(null);
    clearErrors();
    setServerError('');
  };

  const nextStep = async () => {
    let isValid = false;
    
    switch(currentStep) {
      case 1:
        isValid = await trigger(['teacherName', 'gender', 'DOB']);
        break;
      case 2:
        isValid = await trigger(['teacherEmail', 'teacherContact', 'emergencyContact']);
        break;
      case 3:
        isValid = await trigger(['street', 'country', 'state', 'city', 'pin']);
        break;
      case 4:
        isValid = await trigger(['qualification', 'role', 'onboardingDate']);
        break;
      case 5:
        isValid = true; // Documents are optional
        break;
    }

    if (isValid) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const customStyles = {
    control: (base, state) => ({
      ...base,
      border: state.isFocused ? '2px solid #4f46e5' : errors.country || errors.state || errors.city || errors.pin || errors.role || errors.qualification ? '1px solid #ef4444' : '1px solid #d1d5db',
      boxShadow: state.isFocused ? '0 0 0 3px rgba(79, 70, 229, 0.1)' : 'none',
      minHeight: '48px',
      '&:hover': {
        borderColor: state.isFocused ? '#4f46e5' : '#9ca3af'
      }
    }),
    option: (base, state) => ({
      ...base,
      backgroundColor: state.isSelected ? '#4f46e5' : state.isFocused ? '#e5e7eb' : 'white',
      color: state.isSelected ? 'white' : '#374151',
      padding: '10px 12px',
      '&:active': {
        backgroundColor: '#4f46e5',
        color: 'white'
      }
    }),
    menu: (base) => ({
      ...base,
      zIndex: 9999
    }),
    placeholder: (base) => ({
      ...base,
      color: '#9ca3af'
    })
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-8 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-t-2xl shadow-lg p-6 border-b-4 border-indigo-600">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-3 rounded-xl">
              <UserPlus className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Add New Teacher</h1>
              <p className="text-gray-600 mt-1">Complete the form in {steps.length} simple steps</p>
            </div>
          </div>

          {/* Progress Steps */}
          <div className="mt-6">
            <div className="flex items-center justify-between">
              {steps.map((step, index) => (
                <React.Fragment key={step.id}>
                  <div className="flex flex-col items-center">
                    <div className={`
                      w-10 h-10 rounded-full flex items-center justify-center
                      ${currentStep >= step.id 
                        ? 'bg-indigo-600 text-white border-2 border-indigo-600' 
                        : 'bg-gray-100 text-gray-400 border-2 border-gray-300'
                      }
                      ${currentStep === step.id ? 'ring-4 ring-indigo-100' : ''}
                    `}>
                      {step.icon}
                    </div>
                    <span className={`
                      mt-2 text-xs font-medium
                      ${currentStep >= step.id ? 'text-indigo-600' : 'text-gray-500'}
                    `}>
                      {step.name}
                    </span>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`
                      flex-1 h-1 mx-2
                      ${currentStep > step.id ? 'bg-indigo-600' : 'bg-gray-300'}
                    `} />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Success Message */}
        {submitSuccess && (
          <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-6 rounded-r-lg shadow-md">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800">
                  Teacher registered successfully!
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} encType='multipart/form-data' className="bg-white rounded-b-2xl shadow-lg p-8">
          {currentStep === 1 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <UserPlus className="w-5 h-5 text-indigo-600" />
                Personal Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Full Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    {...register('teacherName', {
                      required: 'Teacher name is required',
                      minLength: { value: 2, message: 'Minimum 2 characters required' },
                      pattern: { value: /^[A-Za-z\s]+$/, message: 'Only letters allowed' }
                    })}
                    className={`w-full px-4 py-3 border ${errors.teacherName ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                    placeholder="Enter full name"
                  />
                  {errors.teacherName && (
                    <p className="mt-1 text-sm text-red-500">{errors.teacherName.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender <span className="text-red-500">*</span>
                  </label>
                  <select
                    {...register('gender', { required: 'Gender is required' })}
                    className={`w-full px-4 py-4 border ${errors.gender ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                  >
                    <option value="">Select gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                  {errors.gender && (
                    <p className="mt-1 text-sm text-red-500">{errors.gender.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Date of Birth <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <input
                      type="date"
                      {...register("DOB", {
                        required: "Date of birth is required",
                        validate: (value) => {
                          const birthDate = new Date(value);
                          const today = new Date();

                          const age =
                            today.getFullYear() - birthDate.getFullYear() -
                            (today < new Date(today.getFullYear(), birthDate.getMonth(), birthDate.getDate()) ? 1 : 0);

                          if (age < 21) return "Teacher must be at least 21 years old";
                          if (age > 70) return "Teacher must be less than 70 years old";
                          return true;
                        },
                      })}
                      min={new Date(new Date().setFullYear(new Date().getFullYear() - 70))
                        .toISOString()
                        .split("T")[0]}
                      max={new Date(new Date().setFullYear(new Date().getFullYear() - 21))
                        .toISOString()
                        .split("T")[0]}
                      className={`w-full pl-10 pr-4 py-3 border ${
                        errors.DOB ? "border-red-500" : "border-gray-300"
                      } rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                    />
                  </div>
                  {errors.DOB && (
                    <p className="mt-1 text-sm text-red-500">{errors.DOB.message}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <Mail className="w-5 h-5 text-indigo-600" />
                Contact Information
              </h2>
              {serverError && <p className="text-red-500 mb-4">{serverError}</p>}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email Address <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <input
                      type="email"
                      {...register('teacherEmail', {
                        required: 'Email is required',
                        pattern: {
                          value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                          message: 'Invalid email address'
                        }
                      })}
                      onChange={(e) => handleEmail(e)}
                      className={`w-full pl-10 pr-4 py-3 border ${errors.teacherEmail ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      placeholder="teacher@school.com"
                    />
                  </div>
                  {errors.teacherEmail && (
                    <p className="mt-1 text-sm text-red-500">{errors.teacherEmail.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Primary Phone <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <input
                      type="tel"
                      {...register('teacherContact', {
                        required: 'Primary phone is required',
                        pattern: { value: /^[0-9]{10}$/, message: 'Must be 10 digits' }
                      })}
                      className={`w-full pl-10 pr-4 py-3 border ${errors.teacherContact ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      placeholder="9876543210"
                    />
                  </div>
                  {errors.teacherContact && (
                    <p className="mt-1 text-sm text-red-500">{errors.teacherContact.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Secondary Phone
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <input
                      type="tel"
                      {...register('emergencyContact', {
                        pattern: { value: /^[0-9]{10}$/, message: 'Must be 10 digits' }
                      })}
                      className={`w-full pl-10 pr-4 py-3 border ${errors.emergencyContact ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      placeholder="9876543210"
                    />
                  </div>
                  {errors.emergencyContact && (
                    <p className="mt-1 text-sm text-red-500">{errors.emergencyContact.message}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {currentStep === 3 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <MapPin className="w-5 h-5 text-indigo-600" />
                Address Details
              </h2>
              <div className="grid grid-cols-1 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Street Address <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Home className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <textarea
                      {...register('address', {
                        required: 'Street address is required',
                        minLength: { value: 10, message: 'Address too short' }
                      })}
                      rows="2"
                      className={`w-full pl-10 pr-4 py-3 border ${errors.address ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      placeholder="House no., Building, Street, Area"
                    />
                  </div>
                  {errors.street && (
                    <p className="mt-1 text-sm text-red-500">{errors.address.message}</p>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Country <span className="text-red-500">*</span>
                    </label>
                    <Select
                      options={countries}
                      value={selectedCountry}
                      onChange={(option) => {
                        setSelectedCountry(option);
                        setSelectedState(null);
                        setSelectedCity(null);
                        setSelectedPin(null);
                        setValue('country', option?.value || '');
                        setValue('state', '');
                        setValue('city', '');
                        setValue('pin', '');
                        clearErrors(['state', 'city', 'pin']);
                      }}
                      placeholder="Search country..."
                      isSearchable
                      styles={customStyles}
                      className="react-select-container"
                      classNamePrefix="react-select"
                      required
                    />
                    <input
                      type="hidden"
                      {...register('country', { required: 'Country is required' })}
                    />
                    {errors.country && (
                      <p className="mt-1 text-sm text-red-500">{errors.country.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      State <span className="text-red-500">*</span>
                    </label>
                    <AsyncSelect
                      cacheOptions
                      defaultOptions
                      loadOptions={loadStates}
                      value={selectedState}
                      onChange={(option) => {
                        setSelectedState(option);
                        setSelectedCity(null);
                        setSelectedPin(null);
                        setValue('state', option?.value || '');
                        setValue('city', '');
                        setValue('pin', '');
                        clearErrors(['city', 'pin']);
                      }}
                      placeholder="Search state..."
                      isDisabled={!selectedCountry}
                      isSearchable
                      styles={customStyles}
                      className="react-select-container"
                      classNamePrefix="react-select"
                      loadingMessage={() => "Loading states..."}
                      noOptionsMessage={() => selectedCountry ? "Type to search states..." : "Select country first"}
                      required
                    />
                    <input
                      type="hidden"
                      {...register('state', { required: 'State is required' })}
                    />
                    {errors.state && (
                      <p className="mt-1 text-sm text-red-500">{errors.state.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      City <span className="text-red-500">*</span>
                    </label>
                    <AsyncSelect
                      cacheOptions
                      defaultOptions
                      loadOptions={loadCities}
                      value={selectedCity}
                      onChange={(option) => {
                        setSelectedCity(option);
                        setSelectedPin(null);
                        setValue('city', option?.value || '');
                        setValue('pin', '');
                        clearErrors(['pin']);
                      }}
                      placeholder="Search city..."
                      isDisabled={!selectedState}
                      isSearchable
                      styles={customStyles}
                      className="react-select-container"
                      classNamePrefix="react-select"
                      loadingMessage={() => "Loading cities..."}
                      noOptionsMessage={() => selectedState ? "Type to search cities..." : "Select state first"}
                      required
                    />
                    <input
                      type="hidden"
                      {...register('city', { required: 'City is required' })}
                    />
                    {errors.city && (
                      <p className="mt-1 text-sm text-red-500">{errors.city.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      PIN Code <span className="text-red-500">*</span>
                    </label>
                    <AsyncSelect
                      cacheOptions
                      defaultOptions
                      loadOptions={loadPincodes}
                      value={selectedPin}
                      onChange={(option) => {
                        setSelectedPin(option);
                        setValue('pin', option?.value || '');
                      }}
                      placeholder="Search PIN code..."
                      isDisabled={!selectedCity}
                      isSearchable
                      styles={customStyles}
                      className="react-select-container"
                      classNamePrefix="react-select"
                      loadingMessage={() => "Loading PIN codes..."}
                      noOptionsMessage={() => selectedCity ? "Type to search PIN codes..." : "Select city first"}
                      required
                    />
                    <input
                      type="hidden"
                      {...register('pin', { required: 'PIN code is required' })}
                    />
                    {errors.pin && (
                      <p className="mt-1 text-sm text-red-500">{errors.pin.message}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 4 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-indigo-600" />
                Professional Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Highest Qualification <span className="text-red-500">*</span>
                  </label>
                  <Select
                    options={qualificationOptions}
                    value={selectedQualification}
                    onChange={(option) => {
                      setSelectedQualification(option);
                      setValue('qualification', option?.value || '');
                    }}
                    placeholder="Search qualification..."
                    isSearchable
                    styles={customStyles}
                    className="react-select-container"
                    classNamePrefix="react-select"
                    required
                  />
                  <input
                    type="hidden"
                    {...register('qualification', { required: 'Qualification is required' })}
                  />
                  {errors.qualification && (
                    <p className="mt-1 text-sm text-red-500">{errors.qualification.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Role <span className="text-red-500">*</span>
                  </label>
                  <Select
                    options={roleOptions}
                    value={selectedRole}
                    onChange={(option) => {
                      setSelectedRole(option);
                      setValue('role', option?.value || '');
                    }}
                    placeholder="Search role..."
                    isSearchable
                    styles={customStyles}
                    className="react-select-container"
                    classNamePrefix="react-select"
                    required
                  />
                  <input
                    type="hidden"
                    {...register('role', { required: 'Role is required' })}
                  />
                  {errors.role && (
                    <p className="mt-1 text-sm text-red-500">{errors.role.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Joining Date <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <input
                      type="date"
                      {...register('onboardingDate', {
                        required: 'Joining date is required'
                      })}
                      className={`w-full pl-10 pr-4 py-3 border ${errors.onboardingDate ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                    />
                  </div>
                  {errors.onboardingDate && (
                    <p className="mt-1 text-sm text-red-500">{errors.onboardingDate.message}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {currentStep === 5 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-600" />
                Necessary Documents (Optional)
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Passport Size Photo
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    {...register('photo')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    PAN Card
                  </label>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('PAN')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Aadhar Card
                  </label>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('aadhar')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Address Proof
                  </label>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('addressProof')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Driving License
                  </label>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('DL')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between pt-6 border-t border-gray-200">
            <div>
              {currentStep > 1 && (
                <button
                  type="button"
                  onClick={prevStep}
                  className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 focus:ring-4 focus:ring-gray-200 transition flex items-center justify-center gap-2"
                >
                  <ChevronLeft className="w-5 h-5" />
                  Previous
                </button>
              )}
            </div>

            <div className="flex gap-4">
              {currentStep < steps.length ? (
                <button
                  type="button"
                  onClick={nextStep}
                  className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 focus:ring-4 focus:ring-indigo-300 transition flex items-center justify-center gap-2"
                >
                  Next
                  <ChevronRight className="w-5 h-5" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 focus:ring-4 focus:ring-green-300 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <Save className="w-5 h-5" />
                      Submit Teacher Registration
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Cancel Button (only shown on last step) */}
          {currentStep === steps.length && (
            <div className="mt-4 flex justify-center">
              <button
                type="button"
                onClick={handleReset}
                disabled={isSubmitting}
                className="px-6 py-3 border-2 border-red-300 text-red-700 rounded-lg font-semibold hover:bg-red-50 focus:ring-4 focus:ring-red-200 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <X className="w-5 h-5" />
                Cancel Registration
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}