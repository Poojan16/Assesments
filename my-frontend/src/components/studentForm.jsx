import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  UserPlus, Mail, Phone, MapPin, Calendar, BookOpen, Save, X, Upload,
  ChevronRight, ChevronLeft, Globe, Home, FileText, Users, GraduationCap,
  Edit2,
  Edit3
} from 'lucide-react';
import axios from 'axios';
import Select from 'react-select';
import AsyncSelect from 'react-select/async';
import { useDispatch, useSelector } from 'react-redux';
import { initializeAuth } from '../authSlice';

export default function StudentForm() {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEditMode = !!id;
  
  const {user} = useSelector((state) => state.auth);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(initializeAuth());
  }, [dispatch]);

  const [teacher, setTeacher] = useState(null);

  useEffect(() => {
    const fetchTeachers = async (user) => {
      try {
        const response = await fetch('http://127.0.0.1:8000/teachers/email?email=' + user?.userEmail);
        const data = await response.json();
        setTeacher(data?.data);
      } catch (error) {
        console.error('Error fetching teachers:', error);
      }
    };
    fetchTeachers(user);
  }, [user]);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [classes, setClasses] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [currentStep, setCurrentStep] = useState(1);
  const [countries, setCountries] = useState([{ value: 'India', label: 'India' }]); // Default to India
  const [selectedCountry, setSelectedCountry] = useState({ value: 'India', label: 'India' });
  const [selectedState, setSelectedState] = useState(null);
  const [selectedCity, setSelectedCity] = useState(null);
  const [selectedPin, setSelectedPin] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);
  const [selectedTeacher, setSelectedTeacher] = useState(null);
  const [showParentInfo, setShowParentInfo] = useState(isEditMode ? false : true); // Show by default for edit
  const [existingPhotos, setExistingPhotos] = useState({
    photo: '',
    adhaar: '', // Changed from aadhar to adhaar
    birthCertificate: '',
    parentAadhar: ''
  });
  const [isLoading, setIsLoading] = useState(isEditMode);
  const [studentData, setStudentData] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
    setValue,
    trigger,
    clearErrors,
  } = useForm({
    defaultValues: {
      studentName: '',
      DOB: '',
      gender: '',
      classId: '',
      schoolId: '',
      photo: null,
      adhaar: null, // Changed from aadhar to adhaar
      birthCertificate: null,
      teacherId: '',
      country: 'India',
      state: '',
      city: '',
      pin: '',
      address: '',
      parentName: '',
      parentEmail: '',
      parentContact: '',
      parentRelation: '',
      parentAadhar: null,
    }
  });

  // FIX: Update steps array to handle edit mode properly
  const steps = isEditMode ? [
    { id: 1, name: 'Personal Info', icon: <UserPlus className="w-4 h-4" /> },
    { id: 2, name: 'Address', icon: <MapPin className="w-4 h-4" /> },
    { id: 3, name: 'Academic', icon: <GraduationCap className="w-4 h-4" /> },
    { id: 4, name: 'Documents', icon: <FileText className="w-4 h-4" /> },
  ] : [
    { id: 1, name: 'Personal Info', icon: <UserPlus className="w-4 h-4" /> },
    { id: 2, name: 'Address', icon: <MapPin className="w-4 h-4" /> },
    { id: 3, name: 'Academic', icon: <GraduationCap className="w-4 h-4" /> },
    { id: 4, name: 'Documents', icon: <FileText className="w-4 h-4" /> },
    { id: 5, name: 'Parent', icon: <Users className="w-4 h-4" /> },
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

  // Load states for selected country
  const loadStates = async (inputValue) => {
    try {
      const response = await axios.post('https://countriesnow.space/api/v0.1/countries/states', {
        country: "India"
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
        country: "India",
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
      const listCities = ['Bengaluru']
      const city  = selectedCity.value
      if(listCities.includes(city)){
        city = 'Bangalore'
      }
      const response = await axios.get(`https://api.postalpincode.in/postoffice/${city}`);
      console.log(response);
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

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [classesResponse, teachersResponse] = await Promise.all([
          fetch('http://127.0.0.1:8000/classes/'),
          fetch('http://127.0.0.1:8000/teachers/')
        ]);

        if (!classesResponse.ok) throw new Error(`HTTP error! status: ${classesResponse.status}`);
        if (!teachersResponse.ok) throw new Error(`HTTP error! status: ${teachersResponse.status}`);

        const classesData = await classesResponse.json();
        const teachersData = await teachersResponse.json();

        setClasses(classesData?.data);
        setTeachers(teachersData?.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
    
    if (isEditMode) {
      fetchStudentData();
    }
  }, [id]);

  // Format classes for react-select
  const classOptions = classes.map(cls => ({
    value: cls.classId,
    label: `${cls.className}`
  }));

  // Format teachers for react-select
  const teacherOptions = teachers.map(teacher => ({
    value: teacher.teacherId,
    label: `${teacher.teacherName} - ${teacher.qualification || 'No Qualification'}`
  }));

  // Fetch student data for edit mode - FIXED
  const fetchStudentData = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`http://127.0.0.1:8000/students/id?studentId=${id}`);
      if (!response.ok) throw new Error('Failed to fetch student data');
      
      const result = await response.json();
      const data = result?.data || {};
      
      setStudentData(data);
      
      // Set form values
      Object.keys(data).forEach(key => {
        if (key in watch()) { // Only set if field exists in form
          // Handle null values
          const value = data[key];
          if (value !== null && value !== undefined) {
            setValue(key, value);
          }
        }
      });

      // FIX: Set existing file URLs with correct field names
      setExistingPhotos({
        photo: data.photo || '',
        adhaar: data.adhaar || '', // Changed from aadhar to adhaar
        birthCertificate: data.birthCertificate || '',
        parentAadhar: data.parentAadhar || ''
      });

      // Set selected country (default to India)
      setSelectedCountry({ value: 'India', label: 'India' });
      
      // Load and set state
      if (data.state) {
        const statesData = await loadStates('');
        const stateOption = statesData.find(s => s.value === data.state);
        if (stateOption) {
          setSelectedState(stateOption);
          
          // Load and set city
          if (data.city) {
            console.log(data.city);
            const citiesData = await loadCities('');
            console.log(citiesData);
            const cityOption = citiesData.find(c => c.value === data.city);
            if (cityOption) {
              setSelectedCity(cityOption);

              // Load and set pin
              if (data.pin) {
                console.log(data.pin);
                const pincodesData = await loadPincodes('');
                console.log(pincodesData);
                const pinOption = pincodesData.find(p => p.value.toString() === data.pin.toString());
                if (pinOption) {
                  setSelectedPin(pinOption);
                }
              }
            }
            
          }
        }
      }

      // Set class
      if (data.classId) {
        const classOption = classOptions.find(c => c.value === data.classId);
        setSelectedClass(classOption);
      }

      // Set teacher
      if (data.teacherId) {
        const teacherOption = teacherOptions.find(t => t.value === data.teacherId);
        setSelectedTeacher(teacherOption);
      }

      // Set parent info toggle based on existing data
      if (data.parentName || data.parentEmail || data.parentContact) {
        setShowParentInfo(true);
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching student data:', error);
      setIsLoading(false);
    }
  };

  const onSubmit = async (data) => {
    setIsSubmitting(true);

    const form = document.querySelector('form');
    const formElements = form.elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = true;
    }

    console.log(teacher)

    const formData = new FormData();
    
    // Add all fields to formData
    Object.keys(data).forEach(key => {
      if (key === "photo" || key === "adhaar" || key === "birthCertificate" || key === "parentAadhar") {
        // Handle file fields
        if (data[key] instanceof FileList) {
          if (data[key].length > 0) {
            formData.append(key, data[key][0]);
          }
        } else if (data[key] === '') {
          // Handle file removal in edit mode
          formData.append(key, '');
        }
      } else if (key === 'schoolId') {
        formData.append(key, teacher?.schoolId || '');
      } else if (data[key] !== null && data[key] !== undefined && data[key] !== "") {
        formData.append(key, data[key]);
      }
    });

    // For edit mode, add studentId
    if (isEditMode) {
      formData.append('studentId', id);
    }

    const url = isEditMode 
      ? `http://127.0.0.1:8000/students/id`
      : 'http://127.0.0.1:8000/students/';
    
    const method = isEditMode ? 'PUT' : 'POST';
    
    try {
      const response = await fetch(url, {
        method: method,
        body: formData
      });
      
      if (response.ok) {
        setSubmitSuccess(true);
        await response.json();
        
        setTimeout(() => {
          setSubmitSuccess(false);
          if (isEditMode) {
            // Refresh the data
            fetchStudentData();
            // Re-enable form
            for (let i = 0; i < formElements.length; i++) {
              formElements[i].disabled = false;
            }
          } else {
            handleReset();
          }
        }, 3000);
      } else {
        console.error('Failed to save student');
        // Re-enable form on error
        for (let i = 0; i < formElements.length; i++) {
          formElements[i].disabled = false;
        }
      }
    } catch (error) {
      console.error('Error saving student:', error);
      // Re-enable form on error
      for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = false;
      }
    }
    
    setIsSubmitting(false);
  };

  const handleReset = () => {
    reset();
    setSubmitSuccess(false);
    setCurrentStep(1);
    setSelectedCountry({ value: 'India', label: 'India' });
    setSelectedState(null);
    setSelectedCity(null);
    setSelectedPin(null);
    setSelectedClass(null);
    setSelectedTeacher(null);
    setShowParentInfo(!isEditMode);
    setExistingPhotos({
      photo: '',
      adhaar: '',
      birthCertificate: '',
      parentAadhar: ''
    });
    clearErrors();
    
    if (!isEditMode) {
      navigate('/students');
    } else {
      // For edit mode, reset to original data
      if (studentData) {
        Object.keys(studentData).forEach(key => {
          if (key in watch()) {
            setValue(key, studentData[key]);
          }
        });
      }
    }
  };

  const nextStep = async () => {
    let isValid = false;
    
    switch(currentStep) {
      case 1:
        isValid = await trigger(['studentName', 'gender', 'DOB']);
        break;
      case 2:
        isValid = await trigger(['address', 'state', 'city', 'pin']);
        break;
      case 3:
        isValid = await trigger(['classId', 'teacherId']);
        break;
      case 4:
        isValid = true; // Documents are optional
        break;
      case 5:
        if (showParentInfo) {
          isValid = await trigger(['parentName']);
        } else {
          isValid = true;
        }
        break;
    }

    if (isValid) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const toggleParentInfo = () => {
    setShowParentInfo(!showParentInfo);
    if (!showParentInfo) {
      setValue('parentName', '');
      setValue('parentEmail', '');
      setValue('parentContact', '');
      setValue('parentRelation', '');
      setValue('parentAadhar', null);
    }
  };

  const customStyles = {
    control: (base, state) => ({
      ...base,
      border: state.isFocused ? '2px solid #4f46e5' : errors.country || errors.state || errors.city || errors.pin ? '1px solid #ef4444' : '1px solid #d1d5db',
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

  // Handle file removal
  const handleFileRemoval = (fieldName) => {
    setValue(fieldName, '');
    setExistingPhotos(prev => ({
      ...prev,
      [fieldName]: ''
    }));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-gray-700">Loading student data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-t-2xl shadow-lg p-6 border-b-4 border-indigo-600">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-3 rounded-xl ${isEditMode ? 'bg-yellow-600' : 'bg-indigo-600'}`}>
                {isEditMode ? (
                  <Edit2 className="w-8 h-8 text-white" />
                ) : (
                  <UserPlus className="w-8 h-8 text-white" />
                )}
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">
                  {isEditMode ? 'Edit Student' : 'Add New Student'}
                </h1>
                <p className="text-gray-600 mt-1">
                  {isEditMode ? 'Update student information' : `Complete the form in ${steps.length} simple steps`}
                </p>
              </div>
            </div>
            {isEditMode && (
              <div className="bg-yellow-100 text-yellow-800 px-4 py-2 rounded-lg font-medium">
                Editing Mode
              </div>
            )}
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
                        ? (isEditMode ? 'bg-yellow-600' : 'bg-indigo-600') 
                        : 'bg-gray-100 text-gray-400 border-2 border-gray-300'
                      }
                      ${currentStep === step.id ? (isEditMode ? 'ring-4 ring-yellow-100' : 'ring-4 ring-indigo-100') : ''}
                      ${currentStep >= step.id ? 'text-white border-2 ' + (isEditMode ? 'border-yellow-600' : 'border-indigo-600') : ''}
                    `}>
                      {step.icon}
                    </div>
                    <span className={`
                      mt-2 text-xs font-medium
                      ${currentStep >= step.id ? (isEditMode ? 'text-yellow-600' : 'text-indigo-600') : 'text-gray-500'}
                    `}>
                      {step.name}
                    </span>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`
                      flex-1 h-1 mx-2
                      ${currentStep > step.id ? (isEditMode ? 'bg-yellow-600' : 'bg-indigo-600') : 'bg-gray-300'}
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
                  Student {isEditMode ? 'updated' : 'registered'} successfully!
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} encType="multipart/form-data" className="bg-white rounded-b-2xl shadow-lg p-8">
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
                    maxLength={50}
                    disabled={isSubmitting}
                    {...register('studentName', {
                      required: 'Student name is required',
                      minLength: { value: 2, message: 'Minimum 2 characters required' },
                      pattern: { value: /^[A-Za-z\s]+$/, message: 'Only letters allowed' }
                    })}
                    className={`w-full px-4 py-3 border ${errors.studentName ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                    placeholder="Enter full name"
                  />
                  {errors.studentName && (
                    <p className="mt-1 text-sm text-red-500">{errors.studentName.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender <span className="text-red-500">*</span>
                  </label>
                  <select
                    disabled={isSubmitting}
                    {...register('gender', { required: 'Gender is required' })}
                    className={`w-full bg-white px-4 py-4 border ${errors.gender ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
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
                      disabled={isSubmitting}
                      type="date"
                      {...register('DOB', {
                        required: 'Date of birth is required',
                        validate: value => {
                          if (!value) return true;
                          const birthDate = new Date(value);
                          const today = new Date();
                          const age = today.getFullYear() - birthDate.getFullYear();
                          
                          if (age < 5) return 'Student must be at least 5 years old';
                          if (age > 21) return 'Student must be less than 21 years old';
                          return true;
                        }
                      })}
                      className={`w-full pl-10 pr-4 py-3 border ${errors.DOB ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      max={new Date(new Date().setFullYear(new Date().getFullYear() - 5))
                        .toISOString()
                        .split("T")[0]}
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
                    maxLength={255}
                      disabled={isSubmitting}
                      {...register('address', {
                        required: 'Street address is required',
                        minLength: { value: 10, message: 'Address too short' }
                      })}
                      rows="2"
                      className={`w-full pl-10 pr-4 py-3 border ${errors.address ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                      placeholder="House no., Building, Street, Area"
                    />
                  </div>
                  {errors.address && (
                    <p className="mt-1 text-sm text-red-500">{errors.address.message}</p>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Country <span className="text-red-500">*</span>
                    </label>
                    <Select
                      disabled={isSubmitting}
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
                      isDisabled={true} // Country is fixed to India
                    />
                    <input
                      disabled={isSubmitting}
                      type="hidden"
                      {...register('country', { required: 'Country is required' })}
                    />
                    {errors.country && (
                      <p className="mt-1 text-sm text-red-500">{errors.country.message}</p>
                    )}
                  </div> */}

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
                      isDisabled={!selectedCountry || isSubmitting}
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
                      disabled={isSubmitting}
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
                      isDisabled={!selectedState || isSubmitting}
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
                      disabled={isSubmitting}
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
                      isDisabled={!selectedCity || isSubmitting}
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
                      disabled={isSubmitting}
                      {...register('pin', { 
                        required: 'PIN code is required',
                        pattern: {
                          value: /^\d{6}$/,
                          message: 'PIN code must be 6 digits'
                        }
                      })}
                    />
                    {errors.pin && (
                      <p className="mt-1 text-sm text-red-500">{errors.pin.message}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 3 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <GraduationCap className="w-5 h-5 text-indigo-600" />
                Academic Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Class <span className="text-red-500">*</span>
                  </label>
                  <Select
                    disabled={isSubmitting}
                    options={classOptions}
                    value={selectedClass}
                    onChange={(option) => {
                      setSelectedClass(option);
                      setValue('classId', option?.value || '');
                    }}
                    placeholder="Search class..."
                    isSearchable
                    styles={customStyles}
                    className="react-select-container"
                    classNamePrefix="react-select"
                    required
                  />
                  <input
                    type="hidden"
                    disabled={isSubmitting}
                    {...register('classId', { required: 'Class is required' })}
                  />
                  {errors.classId && (
                    <p className="mt-1 text-sm text-red-500">{errors.classId.message}</p>
                  )}
                </div>

                {/* Teacher selection commented out as per your code */}
              </div>
            </div>
          )}

          {currentStep === 4 && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-600" />
                Student Documents {!isEditMode && '(Optional)'}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Passport Size Photo
                  </label>
                  {existingPhotos.photo && (
                    <div className="mb-3">
                      <div className="relative inline-block">
                        <img 
                          src={`data:image/jpeg;base64,${existingPhotos.photo}`} 
                          alt="Current photo" 
                          className="w-20 h-20 object-cover rounded-lg border"
                        />
                        <button
                          type="button"
                          onClick={() => handleFileRemoval('photo')}
                          disabled={isSubmitting}
                          className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 disabled:opacity-50"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Current photo</p>
                    </div>
                  )}
                  <input
                    disabled={isSubmitting}
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('photo')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Aadhar Card
                  </label>
                  {existingPhotos.adhaar && (
                    <div className="mb-3">
                      <div className="relative inline-block">
                      <img 
                          src={`data:image/jpeg;base64,${existingPhotos.adhaar}`} 
                          alt="Current photo" 
                          className="w-20 h-20 object-cover rounded-lg border"
                        />
                        <button
                          type="button"
                          onClick={() => handleFileRemoval('adhaar')}
                          disabled={isSubmitting}
                          className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 disabled:opacity-50"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Current file</p>
                    </div>
                  )}
                  <input
                    disabled={isSubmitting}
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('adhaar')} // Changed from aadhar to adhaar
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Birth Certificate
                  </label>
                  {existingPhotos.birthCertificate && (
                    <div className="mb-3">
                      <div className="relative inline-block">
                      <img 
                          src={`data:image/jpeg;base64,${existingPhotos.birthCertificate}`} 
                          alt="Current photo" 
                          className="w-20 h-20 object-cover rounded-lg border"
                        />
                        <button
                          type="button"
                          onClick={() => handleFileRemoval('birthCertificate')}
                          disabled={isSubmitting}
                          className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 disabled:opacity-50"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Current file</p>
                    </div>
                  )}
                  <input
                    disabled={isSubmitting}
                    type="file"
                    accept=".jpg,.jpeg,.png,.pdf"
                    {...register('birthCertificate')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  />
                  <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                </div>
              </div>
            </div>
          )}

          {/* Parent Info - Only show for add mode */}
          {!isEditMode && currentStep === 5 && (
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                  <Users className="w-5 h-5 text-indigo-600" />
                  Parent/Guardian Information
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600">Add parent info?</span>
                  <button
                    type="button"
                    onClick={toggleParentInfo}
                    disabled={isSubmitting}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full ${showParentInfo ? 'bg-indigo-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${showParentInfo ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
              </div>

              {showParentInfo ? (
                <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Parent/Guardian Name <span className="text-red-500">*</span>
                      </label>
                      <input
                        disabled={isSubmitting}
                        maxLength={50}
                        type="text"
                        {...register('parentName', { 
                          required: showParentInfo ? 'Parent name is required' : false 
                        })}
                        className={`w-full px-4 py-3 border ${errors.parentName ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                        placeholder="Enter parent/guardian name"
                      />
                      {errors.parentName && (
                        <p className="mt-1 text-sm text-red-500">{errors.parentName.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Relation with Student
                      </label>
                      <select
                        disabled={isSubmitting}
                        {...register('parentRelation')}
                        className="w-full px-4 py-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                      >
                        <option value="">Select Relation</option>
                        <option value="father">Father</option>
                        <option value="mother">Mother</option>
                        <option value="guardian">Guardian</option>
                        <option value="grandparent">Grandparent</option>
                        <option value="sibling">Sibling</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Email Address
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                        <input
                          maxLength={255}
                          disabled={isSubmitting}
                          type="email"
                          {...register('parentEmail', {
                            pattern: {
                              value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                              message: 'Invalid email address'
                            }
                          })}
                          className={`w-full pl-10 pr-4 py-3 border ${errors.parentEmail ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                          placeholder="parent@email.com"
                        />
                      </div>
                      {errors.parentEmail && (
                        <p className="mt-1 text-sm text-red-500">{errors.parentEmail.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Phone Number
                      </label>
                      <div className="relative">
                        <Phone className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                        <input
                        maxLength={10}
                          disabled={isSubmitting}
                          type="tel"
                          {...register('parentContact', {
                            pattern: { value: /^[0-9]{10}$/, message: 'Must be 10 digits' }
                          })}
                          className={`w-full pl-10 pr-4 py-3 border ${errors.parentContact ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                          placeholder="9876543210"
                        />
                      </div>
                      {errors.parentContact && (
                        <p className="mt-1 text-sm text-red-500">{errors.parentContact.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Parent Aadhar Card
                      </label>
                      {existingPhotos.parentAadhar && (
                        <div className="mb-3">
                          <div className="relative inline-block">
                            <div className="w-20 h-20 bg-gray-100 border rounded-lg flex items-center justify-center">
                              <FileText className="w-8 h-8 text-gray-400" />
                            </div>
                            <button
                              type="button"
                              onClick={() => handleFileRemoval('parentAadhar')}
                              disabled={isSubmitting}
                              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 disabled:opacity-50"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                          <p className="text-xs text-gray-500 mt-1">Current file</p>
                        </div>
                      )}
                      <input
                        disabled={isSubmitting}
                        type="file"
                        accept=".jpg,.jpeg,.png,.pdf"
                        {...register('parentAadhar')}
                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                      />
                      <p className="mt-1 text-xs text-gray-500">JPG, PNG, PDF up to 5MB</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-center py-8 bg-gray-50 rounded-lg border border-gray-200">
                    <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">Parent information is not required.</p>
                    <p className="text-sm text-gray-500 mt-1">You can add it later if needed.</p>
                    <p className="text-sm text-gray-500 mt-1">However, please provide parent email address.</p>
                  </div>
                  <div className='mt-4'>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                      <input
                        disabled={isSubmitting}
                        type="email"
                        maxLength={255}
                        {...register('parentEmail', {
                          pattern: {
                            value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                            message: 'Invalid email address'
                          }
                        })}
                        className={`w-full pl-10 pr-4 py-3 border ${errors.parentEmail ? 'border-red-500' : 'border-gray-300'} rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition`}
                        placeholder="parent@email.com"
                      />
                    </div>
                    {errors.parentEmail && (
                      <p className="mt-1 text-sm text-red-500">{errors.parentEmail.message}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between pt-6 border-t border-gray-200">
            <div>
              {currentStep > 1 && (
                <button
                  type="button"
                  onClick={prevStep}
                  disabled={isSubmitting}
                  className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 focus:ring-4 focus:ring-gray-200 transition flex items-center justify-center gap-2 disabled:opacity-50"
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
                  disabled={isSubmitting}
                  className={`px-6 py-3 rounded-lg font-semibold focus:ring-4 transition flex items-center justify-center gap-2 ${isEditMode ? 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-300' : 'bg-indigo-600 hover:bg-indigo-700 focus:ring-indigo-300'} text-white disabled:opacity-50`}
                >
                  Next
                  <ChevronRight className="w-5 h-5" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className={`flex-1 px-6 py-3 rounded-lg font-semibold hover:bg-opacity-90 focus:ring-4 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 ${isEditMode ? 'bg-yellow-600 focus:ring-yellow-300' : 'bg-green-600 focus:ring-green-300'} text-white`}
                >
                  {isSubmitting ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      {isEditMode ? 'Updating...' : 'Submitting...'}
                    </>
                  ) : (
                    <>
                      <Save className="w-5 h-5" />
                      {isEditMode ? 'Update Student' : 'Submit Student Registration'}
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
                {isEditMode ? 'Cancel Edit' : 'Cancel Registration'}
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}