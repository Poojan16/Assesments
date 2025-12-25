// import React, { useState, useMemo, useEffect, useRef } from 'react';
// import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
// import { ChevronLeft, ChevronRight, Search, X, Filter, Award, Building2, MapPin, BarChart3, Users, School } from 'lucide-react';

// const EducationDashboard = () => {
//   const [selectedView, setSelectedView] = useState('overview');
//   const [selectedState, setSelectedState] = useState('All');
//   const [selectedCity, setSelectedCity] = useState('All');
//   const [selectedMetric, setSelectedMetric] = useState('avgScore');
//   const [searchTerm, setSearchTerm] = useState('');
//   const [currentPage, setCurrentPage] = useState(1);
//   const [itemsPerPage, setItemsPerPage] = useState(10);
//   const [showStateDropdown, setShowStateDropdown] = useState(false);
//   const [showCityDropdown, setShowCityDropdown] = useState(false);
//   const [stateSearch, setStateSearch] = useState('');
//   const [citySearch, setCitySearch] = useState('');
  
//   const stateDropdownRef = useRef(null);
//   const cityDropdownRef = useRef(null);
  
//   const [schools, setSchools] = useState([]);
//   const [teachers, setTeachers] = useState([]);
//   const [students, setStudents] = useState([]);
//   const [subjects, setSubjects] = useState([]);
//   const [scores, setScores] = useState([]);
//   const [grades, setGrades] = useState([]);

//   // Close dropdowns when clicking outside
  

//   // Fetch data
//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const [schoolResponse, teacherResponse, studentResponse, subjectResponse, scoreResponse, gradeResponse] = await Promise.all([
//           fetch('http://127.0.0.1:8000/admin/schools/'),
//           fetch('http://127.0.0.1:8000/teachers/'),
//           fetch('http://127.0.0.1:8000/students/'),
//           fetch('http://127.0.0.1:8000/subjects/'),
//           fetch('http://127.0.0.1:8000/students/score'),
//           fetch('http://127.0.0.1:8000/grades/')
//         ]);

//         const [schoolData, teacherData, studentData, subjectData, scoreData, gradeData] = await Promise.all([
//           schoolResponse.json(),
//           teacherResponse.json(),
//           studentResponse.json(),
//           subjectResponse.json(),
//           scoreResponse.json(),
//           gradeResponse.json()
//         ]);

//         setSchools(schoolData?.data || []);
//         setTeachers(teacherData?.data || []);
//         setStudents(studentData?.data || []);
//         setSubjects(subjectData?.data || []);
//         setScores(scoreData?.data || []);
//         setGrades(gradeData?.data || []);
//       } catch (error) {
//         console.error('Error fetching data:', error);
//       }
//     };

//     fetchData();
//   }, []);

  

//   // Performance category function
//   const getPerformanceCategory = (avgScore) => {
//     if (avgScore >= 85) return 'Excellent';
//     if (avgScore >= 70) return 'Good';
//     if (avgScore >= 60) return 'Average';
//     if (avgScore >= 50) return 'Below Average';
//     return 'Poor';
//   };

//   const CustomTooltip = ({ active, payload }) => {
//     if (active && payload && payload.length) {
//       const data = payload[0].payload;
//       return (
//         <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
//           <p className="font-semibold text-gray-900">{data.state || data.city}</p>
//           <p className="text-sm text-gray-600">Average Score: {data.avgScore?.toFixed(1) || 'N/A'}</p>
//           <p className="text-sm text-gray-600">Schools: {data.schools}</p>
//         </div>
//       );
//     }
//     return null;
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-green-50 p-4 md:p-6">
//       <div className="max-w-7xl mx-auto">
//         {/* Header and Navigation */}
//         <div className="mb-6">
//           <div className="flex flex-wrap gap-2 mt-4">
//             {['overview', 'performance', 'regional', 'subjects'].map(view => (
//               <button
//                 key={view}
//                 onClick={() => setSelectedView(view)}
//                 className={`px-4 py-2 rounded-lg font-medium transition capitalize ${
//                   selectedView === view 
//                     ? 'bg-orange-500 text-white shadow-md' 
//                     : 'bg-white text-gray-700 hover:bg-gray-100'
//                 }`}
//               >
//                 {view}
//               </button>
//             ))}
//           </div>
//         </div>

//         {/* Regional View */}
        

//         {/* Other views... */}
//       </div>
//     </div>
//   );
// };

// export default EducationDashboard;