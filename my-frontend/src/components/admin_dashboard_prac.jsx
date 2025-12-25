import React, { useState, useMemo, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Award, BookOpen, Users, School, GraduationCap, Search, Filter, Download, AlertCircle, Building2, MapPin, BarChart3, X, ChevronLeft, ChevronRight } from 'lucide-react';

const EducationDashboard = () => {
  const [selectedView, setSelectedView] = useState('performance');
  const [searchTerm, setSearchTerm] = useState('');
  const [schools, setSchools] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [students, setStudents] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [scores, setScores] = useState([]);
  const [selectedMetric, setSelectedMetric] = useState('avgScore');
  // sample data for state and city
  const [grades, setGrades] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [showStateDropdown, setShowStateDropdown] = useState(false);
  const [showCityDropdown, setShowCityDropdown] = useState(false);
  const [stateSearch, setStateSearch] = useState('');
  const [citySearch, setCitySearch] = useState('');
  const [selectedCity, setSelectedCity] = useState('All');
  const [selectedState, setSelectedState] = useState('All');
    
  const stateDropdownRef = useRef(null);
  const cityDropdownRef = useRef(null);

  useEffect(() => {
      const handleClickOutside = (event) => {
        if (stateDropdownRef.current && !stateDropdownRef.current.contains(event.target)) {
          setShowStateDropdown(false);
        }
        if (cityDropdownRef.current && !cityDropdownRef.current.contains(event.target)) {
          setShowCityDropdown(false);
        }
      };
  
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);


  useEffect(() => {
    const fetchData = async () => {
      try {
        const [schoolResponse, teacherResponse, studentResponse, subjectResponse, scoreResponse, gradeResponse] = await Promise.all([
          fetch('http://127.0.0.1:8000/admin/schools/'),
          fetch('http://127.0.0.1:8000/teachers/'),
          fetch('http://127.0.0.1:8000/students/'),
          fetch('http://127.0.0.1:8000/subjects/'),
          fetch('http://127.0.0.1:8000/students/score'),
          fetch('http://127.0.0.1:8000/grades/')
        ]);
  
        if (!schoolResponse.ok) throw new Error(`HTTP error! status: ${schoolResponse.status}`);
        if (!teacherResponse.ok) throw new Error(`HTTP error! status: ${teacherResponse.status}`);
        if (!studentResponse.ok) throw new Error(`HTTP error! status: ${studentResponse.status}`);
        if (!subjectResponse.ok) throw new Error(`HTTP error! status: ${subjectResponse.status}`);
        if (!scoreResponse.ok) throw new Error(`HTTP error! status: ${scoreResponse.status}`);
        if (!gradeResponse.ok) throw new Error(`HTTP error! status: ${gradeResponse.status}`);
  
        const [schoolData, teacherData, studentData, subjectData, scoreData, gradeData] = await Promise.all([
          schoolResponse.json(),
          teacherResponse.json(),
          studentResponse.json(),
          subjectResponse.json(),
          scoreResponse.json(),
          gradeResponse.json()
        ]);
  
        setSchools(schoolData?.data || []);
        setTeachers(teacherData?.data || []);
        setStudents(studentData?.data || []);
        setSubjects(subjectData?.data || []);
        setScores(scoreData?.data || []);
        setGrades(gradeData?.data || []);
      } catch (error) {
        console.error('Error fetching schools:', error);
      }
    };
  
    fetchData();
  }, []);

  // studentId -> schoolId
  const studentToSchoolMap = {};
  students.forEach(student => {
    studentToSchoolMap[student.studentId] = student.schoolId;
  });

  // schoolId -> all scores
  const schoolScoresMap = {};
  scores.forEach(score => {
    const schoolId = studentToSchoolMap[score.studentId];
    if (!schoolId) return;

    if (!schoolScoresMap[schoolId]) {
      schoolScoresMap[schoolId] = [];
    }

    schoolScoresMap[schoolId].push(score.score); // numeric score
  });


  const schoolAvgScoreMap = {};
  Object.keys(schoolScoresMap).forEach(schoolId => {
    const s = schoolScoresMap[schoolId];
    schoolAvgScoreMap[schoolId] =
      s.reduce((sum, v) => sum + v, 0) / s.length;
  });


  const schoolsWithAvg = schools.map(school => ({
    ...school,
    avgScore: schoolAvgScoreMap[school.schoolId] || 0
  }));


  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold text-gray-900">{data.name}</p>
          <p className="text-sm text-gray-600">Students: {data.students}</p>
          <p className="text-sm text-gray-600">Avg Score: {data.avgScore}</p>
          <p className="text-sm text-gray-600">Performance: {data.performance}</p>
        </div>
      );
    }
    return null;
  };

  const getPerformanceCategory = (avgScore) => {
    if (avgScore >= 80) return 'High Performance';
    if (avgScore >= 65) return 'Medium Performance';
    return 'Low Performance';
  };

  const scatterData = schoolsWithAvg.map(school => ({
    students: students.filter(s => s.schoolId === school.schoolId).length,
    avgScore: school.avgScore,
    performance: getPerformanceCategory(school.avgScore),
    name: school.schoolName
  }));
  
  
  const sortedSchools = [...schoolsWithAvg].sort(
    (a, b) => b.avgScore - a.avgScore
  );
  
  const topPerformers = sortedSchools.slice(0, 3);
  const bottomPerformers = sortedSchools.slice(-3).reverse();


  // State-wise aggregated data
  const stateData = useMemo(() => {
    const stateMap = {};
    
    schoolsWithAvg.forEach(school => {
      const state = school.state;
      if (!state) return;
      
      if (!stateMap[state]) {
        stateMap[state] = {
          state,
          schools: 0,
          totalAvgScore: 0,
          cities: new Set(),
          totalStudents: 0,
          totalTeachers: 0
        };
      }
      
      stateMap[state].schools++;
      stateMap[state].totalAvgScore += school.avgScore;
      stateMap[state].cities.add(school.city);
      stateMap[state].totalStudents += students.filter(s => s.state === state).length;
      stateMap[state].totalTeachers += teachers.filter(t => t.state === state).length;
    });

    return Object.values(stateMap).map(stateData => ({
      ...stateData,
      cities: stateData.cities.size,
      avgScore: stateData.schools > 0 ? stateData.totalAvgScore / stateData.schools : 0,
      studentCount: stateData.totalStudents,
      teacherCount: stateData.totalTeachers,
    })).sort((a, b) => b.avgScore - a.avgScore);
  }, [schoolsWithAvg]);

  // City-wise aggregated data
  const cityData = useMemo(() => {
    const cityMap = {};
    
    schoolsWithAvg.forEach(school => {
      const city = school.city;
      const state = school.state;
      if (!city || !state) return;
      
      const cityKey = `${city}-${state}`;
      if (!cityMap[cityKey]) {
        cityMap[cityKey] = {
          city,
          state,
          pincode: school.pin,
          schools: 0,
          totalAvgScore: 0,
          totalStudents: 0,
          totalTeachers: 0
        };
      }
      
      cityMap[cityKey].schools++;
      cityMap[cityKey].totalAvgScore += school.avgScore;
      cityMap[cityKey].totalStudents += students.filter(s => s.city === city).length;
      cityMap[cityKey].totalTeachers += teachers.filter(t => t.city === city).length;
    });

    return Object.values(cityMap).map(cityData => ({
      ...cityData,
      avgScore: cityData.schools > 0 ? cityData.totalAvgScore / cityData.schools : 0,
      students: cityData.totalStudents,
      teachers: cityData.totalTeachers,
      studentTeacherRatio: cityData.totalTeachers > 0 
        ? (cityData.totalStudents / cityData.totalTeachers).toFixed(1)
        : 0,
      displayName: `${cityData.city}, ${cityData.state}`
    }));
  }, [schoolsWithAvg]);

  // Get unique states for dropdown
  const uniqueStates = useMemo(() => {
    return ['All', ...new Set(stateData.map(item => item.state).filter(Boolean))];
  }, [stateData]);

  // Get cities based on selected state
  const filteredCities = useMemo(() => {
    const cities = cityData
      .filter(city => {
        if (selectedState === 'All') return true;
        return city.state === selectedState;
      })
      .map(city => city.displayName);
    
    return ['All', ...new Set(cities)];
  }, [cityData, selectedState]);

  // Get city data for selected city
  const getCityData = useMemo(() => {
    if (selectedCity === 'All') {
      return cityData.filter(city => {
        if (selectedState === 'All') return true;
        return city.state === selectedState;
      });
    }
    console.log(selectedCity)
    const [cityName, stateName] = selectedCity.split(', ');

    return cityData.filter(city => 
      city.city === cityName && city.state === stateName
    );
  }, [cityData, selectedState, selectedCity]);

  // Filter and paginate table data
  const tableData = useMemo(() => {
    let data = [...getCityData];
    
    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      data = data.filter(city => 
        city.city.toLowerCase().includes(term) ||
        city.state.toLowerCase().includes(term) 
      );
    }
    
    // Sort by selected metric
    data.sort((a, b) => b[selectedMetric] - a[selectedMetric]);
    
    return data;
  }, [getCityData, searchTerm, selectedMetric]);

  // Pagination calculations
  const totalPages = Math.ceil(tableData.length / itemsPerPage);
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return tableData.slice(startIndex, endIndex);
  }, [tableData, currentPage, itemsPerPage]);

  // Filter states based on search
  const filteredStates = useMemo(() => {
    if (!stateSearch) return uniqueStates;
    return uniqueStates.filter(state => 
      state.toLowerCase().includes(stateSearch.toLowerCase())
    );
  }, [uniqueStates, stateSearch]);

  // Filter cities based on search
  const filteredCityOptions = useMemo(() => {
    if (!citySearch) return filteredCities;
    return filteredCities.filter(city => 
      city.toLowerCase().includes(citySearch.toLowerCase())
    );
  }, [filteredCities, citySearch]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedState, selectedCity, searchTerm, selectedMetric]);

  // Handle state selection
  const handleStateSelect = (state) => {
    setSelectedState(state);
    if (state !== 'All') {
      const citiesInState = filteredCities.filter(city => {
        if (city === 'All') return true;
        const [, cityState] = city.split(', ');
        return cityState === state;
      });
      if (!citiesInState.includes(selectedCity)) {
        setSelectedCity('All');
      }
    }
    setShowStateDropdown(false);
    setStateSearch('');
  };

  // Handle city selection
  const handleCitySelect = (city) => {
    setSelectedCity(city);
    setShowCityDropdown(false);
    setCitySearch('');
  };


  //subject section
  const subjectPerformance = subjects.map(subject => {
    const subjectScores = scores.filter(score => score.subjectId === subject.subjectId);
    const gradeCounts = subjectScores.reduce((counts, score) => {
      const grade = grades.find(grade => grade.gradeId === score.grade);
      if (grade) {
        counts[grade.gradeLetter] = (counts[grade.gradeLetter] || 0) + 1;
      }
      return counts;
    }, {});
    const avgScore = Math.round((subjectScores.reduce((sum, score) => sum + score.score, 0) / subjectScores.length) * 10) / 100;
  
    return {
      subject: subject.subjectName,
      avgScore,
      ...gradeCounts
    };
  });

  console.log(subjectPerformance);

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-green-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          
          {/* Navigation */}
          <div className="flex flex-wrap gap-2 mt-4">
            {['performance', 'regional', 'subjects'].map(view => (
              <button
                key={view}
                onClick={() => setSelectedView(view)}
                className={`px-4 py-2 rounded-lg font-medium transition capitalize ${
                  selectedView === view 
                    ? 'bg-orange-500 text-white shadow-md' 
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>

        {/* Performance Analysis View */}
        {selectedView === 'performance' && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Top Performers */}
              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <Award className="w-5 h-5 text-yellow-500 mr-2" />
                  Top 10 Performing Schools
                </h3>
                <div className="space-y-3">
                  {topPerformers.map((school, index) => (
                    <div key={school.id} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <div className="flex items-center">
                        <span className="font-bold text-green-700 mr-3">{index + 1}</span>
                        <div>
                          <div className="font-medium text-sm">{school.schoolName}</div>
                          <div className="text-xs text-gray-600">{school.city} • {school.state}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-700">{(school.avgScore).toFixed(2)}</div>
                        <div className="text-xs text-gray-600">{students.filter(student => student.schoolId === school.schoolId).length} students</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom Performers */}
              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  Bottom 10 - Priority Intervention
                </h3>
                <div className="space-y-3">
                  {bottomPerformers.map((school, index) => (
                    <div key={school.id} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                      <div className="flex items-center">
                        <span className="font-bold text-red-700 mr-3">{120 - index}</span>
                        <div>
                          <div className="font-medium text-sm">{school.schoolName}</div>
                          <div className="text-xs text-gray-600">{school.city} • {school.state}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-red-700">{(school.avgScore).toFixed(2)}</div>
                        <div className="text-xs text-gray-600">{students.filter(student => student.schoolId === school.schoolId).length} students</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Scatter Plot - Students vs Performance */}
            {/* <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
              <h3 className="text-lg font-bold text-gray-900 mb-4">School Size vs Performance Analysis</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid />
                  <XAxis type="number" dataKey="students" name="Students" unit=" students" />
                  <YAxis type="number" dataKey="avgScore" name="Score" domain={[40, 100]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter name="Government" data={scatterData.filter(d => d.type === 'Government')} fill="#10b981" />
                  <Scatter name="Private" data={scatterData.filter(d => d.type === 'Private')} fill="#3b82f6" />
                  <Scatter name="Govt-Aided" data={scatterData.filter(d => d.type === 'Government-Aided')} fill="#f59e0b" />
                </ScatterChart>
              </ResponsiveContainer>
            </div> */}
            <ResponsiveContainer width="100%" height={500}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  type="number" 
                  dataKey="students" 
                  name="Students" 
                  label={{ value: 'Number of Students', position: 'insideBottom', offset: -10 }}
                  stroke="#6b7280"
                />
                <YAxis 
                  type="number" 
                  dataKey="avgScore" 
                  name="Score" 
                  domain={[40, 100]}
                  label={{ value: 'Average Score', angle: -90, position: 'insideLeft' }}
                  stroke="#6b7280"
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  verticalAlign="top" 
                  height={36}
                  wrapperStyle={{ paddingBottom: '20px' }}
                />
                <Scatter 
                  name="High Performance (≥80)" 
                  data={scatterData.filter(d => d.performance === 'High Performance')} 
                  fill="#10b981" 
                  shape="circle"
                  r={8}
                />
                <Scatter 
                  name="Medium Performance (65-79)" 
                  data={scatterData.filter(d => d.performance === 'Medium Performance')} 
                  fill="#f59e0b"
                  shape="circle"
                  r={8}
                />
                <Scatter 
                  name="Low Performance (<65)" 
                  data={scatterData.filter(d => d.performance === 'Low Performance')} 
                  fill="#ef4444"
                  shape="circle"
                  r={8}
                />
              </ScatterChart>
            </ResponsiveContainer>
            
            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <h3 className="font-semibold text-gray-900">High Performance</h3>
                </div>
                <p className="text-sm text-gray-600">Average Score ≥ 80</p>
                <p className="text-2xl font-bold text-green-600 mt-2">
                  {scatterData.filter(d => d.performance === 'High Performance').length} schools
                </p>
              </div>
    
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                  <h3 className="font-semibold text-gray-900">Medium Performance</h3>
                </div>
                <p className="text-sm text-gray-600">Average Score 65-79</p>
                <p className="text-2xl font-bold text-orange-600 mt-2">
                  {scatterData.filter(d => d.performance === 'Medium Performance').length} schools
                </p>
              </div>
    
              <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <h3 className="font-semibold text-gray-900">Low Performance</h3>
                </div>
                <p className="text-2xl font-bold text-red-600 mt-2">
                  {scatterData.filter(d => d.performance === 'Low Performance').length} schools
                </p>
              </div>
            </div>
          </>
        )}

        {/* State & City Analysis View */}
        {selectedView === 'regional' && (
                  <>
                    {/* Filters Section */}
                    <div className="mb-8 p-6 bg-white rounded-2xl shadow-lg border border-gray-200">
                      <div className="flex flex-col lg:flex-row gap-4 mb-6">
                        <div className="flex-1">
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Search Cities/Schools
                          </label>
                          <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                            <input
                              type="text"
                              placeholder="Search by city, state ..."
                              value={searchTerm}
                              onChange={(e) => setSearchTerm(e.target.value)}
                              className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-200 transition-colors"
                            />
                            {searchTerm && (
                              <button
                                onClick={() => setSearchTerm('')}
                                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                              >
                                <X className="w-5 h-5" />
                              </button>
                            )}
                          </div>
                        </div>
        
                        <div className="flex flex-col sm:flex-row gap-4">
                          <div className="relative flex-1" ref={stateDropdownRef}>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Filter by State
                            </label>
                            <button
                              onClick={() => {
                                setShowStateDropdown(!showStateDropdown);
                                setShowCityDropdown(false);
                              }}
                              className="w-full flex items-center justify-between px-4 py-3 border-2 border-gray-200 rounded-lg hover:border-orange-300 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-200 transition-colors"
                            >
                              <span className={selectedState === 'All' ? 'text-gray-500' : 'text-gray-900'}>
                                {selectedState === 'All' ? 'All States' : selectedState}
                              </span>
                              <Filter className="w-5 h-5 text-gray-400" />
                            </button>
                            
                            {showStateDropdown && (
                              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                                <div className="sticky top-0 bg-white p-2 border-b">
                                  <div className="relative">
                                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <input
                                      type="text"
                                      placeholder="Type to search states..."
                                      value={stateSearch}
                                      onChange={(e) => setStateSearch(e.target.value)}
                                      className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-200"
                                      autoFocus
                                    />
                                  </div>
                                </div>
                                <div className="py-1">
                                  {filteredStates.map((state) => (
                                    <button
                                      key={state}
                                      onClick={() => handleStateSelect(state)}
                                      className={`w-full text-left px-4 py-2 hover:bg-orange-50 transition-colors ${
                                        selectedState === state ? 'bg-orange-50 text-orange-600 font-semibold' : 'text-gray-700'
                                      }`}
                                    >
                                      {state === 'All' ? 'All States' : state}
                                    </button>
                                  ))}
                                  {filteredStates.length === 0 && (
                                    <div className="px-4 py-3 text-gray-500 text-center">
                                      No states found
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
        
                          <div className="relative flex-1" ref={cityDropdownRef}>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Filter by City
                            </label>
                            <button
                              onClick={() => {
                                setShowCityDropdown(!showCityDropdown);
                                setShowStateDropdown(false);
                              }}
                              disabled={selectedState !== 'All' && filteredCities.length === 1}
                              className={`w-full flex items-center justify-between px-4 py-3 border-2 rounded-lg transition-colors ${
                                selectedState !== 'All' && filteredCities.length === 1
                                  ? 'border-gray-100 bg-gray-50 cursor-not-allowed'
                                  : 'border-gray-200 hover:border-orange-300 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-200'
                              }`}
                            >
                              <span className={selectedCity === 'All' ? 'text-gray-500' : 'text-gray-900'}>
                                {selectedCity === 'All' 
                                  ? selectedState === 'All' 
                                    ? 'All Cities' 
                                    : `All Cities in ${selectedState}`
                                  : selectedCity}
                              </span>
                              <Filter className="w-5 h-5 text-gray-400" />
                            </button>
                            
                            {showCityDropdown && selectedState !== 'All' && filteredCities.length > 1 && (
                              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                                <div className="sticky top-0 bg-white p-2 border-b">
                                  <div className="relative">
                                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <input
                                      type="text"
                                      placeholder="Type to search cities..."
                                      value={citySearch}
                                      onChange={(e) => setCitySearch(e.target.value)}
                                      className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-200"
                                      autoFocus
                                    />
                                  </div>
                                </div>
                                <div className="py-1">
                                  {filteredCityOptions.map((city) => (
                                    <button
                                      key={city}
                                      onClick={() => handleCitySelect(city)}
                                      className={`w-full text-left px-4 py-2 hover:bg-orange-50 transition-colors ${
                                        selectedCity === city ? 'bg-orange-50 text-orange-600 font-semibold' : 'text-gray-700'
                                      }`}
                                    >
                                      {city === 'All' 
                                        ? `All Cities in ${selectedState}`
                                        : city}
                                    </button>
                                  ))}
                                  {filteredCityOptions.length === 0 && (
                                    <div className="px-4 py-3 text-gray-500 text-center">
                                      No cities found
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
        
                      {/* Summary Stats */}
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="bg-orange-50 p-4 rounded-xl border border-orange-200">
                          <div className="flex items-center gap-2 mb-1">
                            <MapPin className="w-4 h-4 text-orange-600" />
                            <h3 className="font-semibold text-gray-900">States</h3>
                          </div>
                          <p className="text-2xl font-bold text-orange-600">
                            {selectedState === 'All' ? stateData.length : 1}
                          </p>
                        </div>
                        <div className="bg-green-50 p-4 rounded-xl border border-green-200">
                          <div className="flex items-center gap-2 mb-1">
                            <Building2 className="w-4 h-4 text-green-600" />
                            <h3 className="font-semibold text-gray-900">Cities</h3>
                          </div>
                          <p className="text-2xl font-bold text-green-600">
                            {getCityData.length}
                          </p>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-xl border border-blue-200">
                          <div className="flex items-center gap-2 mb-1">
                            <School className="w-4 h-4 text-blue-600" />
                            <h3 className="font-semibold text-gray-900">Schools</h3>
                          </div>
                          <p className="text-2xl font-bold text-blue-600">
                            {getCityData.reduce((sum, city) => sum + city.schools, 0)}
                          </p>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-xl border border-purple-200">
                          <div className="flex items-center gap-2 mb-1">
                            <Users className="w-4 h-4 text-purple-600" />
                            <h3 className="font-semibold text-gray-900">Students</h3>
                          </div>
                          <p className="text-2xl font-bold text-purple-600">
                            {getCityData.reduce((sum, city) => sum + city.students, 0).toLocaleString() || 0}
                          </p>
                        </div>
                      </div>
                    </div>
        
                    {/* Charts Section */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                      {/* State Comparison Chart */}
                      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                          <BarChart3 className="w-6 h-6 text-orange-600" />
                          State Performance Comparison
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart 
                            data={selectedState === 'All' 
                              ? stateData.slice(0, 8) 
                              : stateData.filter(s => s.state === selectedState)
                            }
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="state" 
                              angle={-45} 
                              textAnchor="end" 
                              height={80} 
                              tick={{ fontSize: 12 }} 
                            />
                            <YAxis />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar 
                              dataKey="avgScore" 
                              name="Average Score" 
                              radius={[8, 8, 0, 0]}
                              fill="#f97316"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
        
                      {/* City Distribution Chart */}
                      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                          <Award className="w-6 h-6 text-orange-600" />
                          Top Cities Performance
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart 
                            data={getCityData
                              .sort((a, b) => b.avgScore - a.avgScore)
                              .slice(0, 8)} 
                            layout="vertical"
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              type="number" 
                              domain={[0, 100]} 
                              tickFormatter={(value) => `${value}%`}
                            />
                            <YAxis 
                              dataKey="city" 
                              type="category" 
                              width={100} 
                              tick={{ fontSize: 11 }} 
                            />
                            <Tooltip 
                              formatter={(value) => [`${value.toFixed(1)}%`, 'Average Score']}
                              labelFormatter={(label) => `City: ${label}`}
                            />
                            <Bar 
                              dataKey="avgScore" 
                              name="Average Score" 
                              radius={[0, 8, 8, 0]}
                              fill="#10b981"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
        
                    {/* Table with Pagination */}
                    <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                            <Award className="w-6 h-6 text-orange-600" />
                            Comprehensive City Rankings
                            <span className="text-sm font-normal text-gray-500 ml-2">
                              ({tableData.length} cities found)
                            </span>
                          </h3>
                          <p className="text-sm text-gray-600 mt-1">
                            Showing {Math.min(itemsPerPage, paginatedData.length)} of {tableData.length} cities
                          </p>
                        </div>
                        
                        <div className="flex flex-col sm:flex-row gap-4">
                          <select
                            value={selectedMetric}
                            onChange={(e) => setSelectedMetric(e.target.value)}
                            className="px-4 py-2 border-2 border-gray-200 rounded-lg font-semibold focus:border-orange-500 focus:outline-none"
                          >
                            <option value="avgScore">By Performance Score</option>
                            <option value="schools">By Number of Schools</option>
                            <option value="students">By Student Count</option>
                            <option value="teachers">By Teacher Count</option>
                          </select>
                          
                          <select
                            value={itemsPerPage}
                            onChange={(e) => {
                              setItemsPerPage(Number(e.target.value));
                              setCurrentPage(1);
                            }}
                            className="px-4 py-2 border-2 border-gray-200 rounded-lg font-semibold focus:border-orange-500 focus:outline-none"
                          >
                            <option value="5">5 per page</option>
                            <option value="10">10 per page</option>
                            <option value="25">25 per page</option>
                            <option value="50">50 per page</option>
                          </select>
                        </div>
                      </div>
                      
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b-2 border-orange-200">
                              <th className="text-left p-3 font-bold text-gray-700">Rank</th>
                              <th className="text-left p-3 font-bold text-gray-700">City</th>
                              <th className="text-left p-3 font-bold text-gray-700">State</th>
                              <th className="text-center p-3 font-bold text-gray-700">Pincode</th>
                              <th className="text-center p-3 font-bold text-gray-700">Score</th>
                              <th className="text-center p-3 font-bold text-gray-700">Schools</th>
                              <th className="text-center p-3 font-bold text-gray-700">Students</th>
                              <th className="text-center p-3 font-bold text-gray-700">Teachers</th>
                              <th className="text-center p-3 font-bold text-gray-700">Ratio</th>
                            </tr>
                          </thead>
                          <tbody>
                            {paginatedData.map((city, index) => {
                              const globalIndex = (currentPage - 1) * itemsPerPage + index;
                              return (
                                <tr 
                                  key={`${city.city}-${city.state}`} 
                                  className={`border-b border-gray-100 hover:bg-orange-50 transition-colors ${globalIndex < 3 ? 'bg-orange-50/50' : ''}`}
                                >
                                  <td className="p-3">
                                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm ${
                                      globalIndex === 0 ? 'bg-yellow-400 text-yellow-900' :
                                      globalIndex === 1 ? 'bg-gray-300 text-gray-700' :
                                      globalIndex === 2 ? 'bg-orange-300 text-orange-900' :
                                      'bg-gray-100 text-gray-600'
                                    }`}>
                                      {globalIndex + 1}
                                    </span>
                                  </td>
                                  <td className="p-3 font-bold text-gray-900">{city.city}</td>
                                  <td className="p-3 text-gray-600">{city.state}</td>
                                  <td className="p-3 text-center text-sm text-gray-500 font-mono">{city.pincode}</td>
                                  <td className="p-3 text-center">
                                    <span className={`inline-block px-3 py-1 rounded-full font-bold text-sm ${
                                      city.avgScore >= 85 ? 'bg-green-100 text-green-700' :
                                      city.avgScore >= 70 ? 'bg-blue-100 text-blue-700' :
                                      city.avgScore >= 60 ? 'bg-yellow-100 text-yellow-700' :
                                      city.avgScore >= 50 ? 'bg-orange-100 text-orange-700' :
                                      'bg-red-100 text-red-700'
                                    }`}>
                                      {city.avgScore.toFixed(1)}%
                                    </span>
                                  </td>
                                  <td className="p-3 text-center font-semibold text-gray-700">{city.schools}</td>
                                  <td className="p-3 text-center font-semibold text-gray-700">{city.students.toLocaleString()}</td>
                                  <td className="p-3 text-center font-semibold text-gray-700">{city.teachers.toLocaleString()}</td>
                                  <td className="p-3 text-center">
                                    <span className="text-sm font-semibold text-blue-600">
                                      {city.studentTeacherRatio}:1
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        
                        {paginatedData.length === 0 && (
                          <div className="text-center py-12 text-gray-500">
                            No cities found matching your search criteria
                          </div>
                        )}
                      </div>
        
                      {/* Pagination Controls */}
                      {tableData.length > 0 && (
                        <div className="flex flex-col sm:flex-row justify-between items-center mt-6 pt-6 border-t border-gray-200">
                          <div className="text-sm text-gray-600 mb-4 sm:mb-0">
                            Showing {Math.min(itemsPerPage, paginatedData.length)} of {tableData.length} results
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                              disabled={currentPage === 1}
                              className={`p-2 rounded-lg border transition-colors ${
                                currentPage === 1
                                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                                  : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                              }`}
                            >
                              <ChevronLeft className="w-5 h-5" />
                            </button>
                            
                            <div className="flex items-center gap-1">
                              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                                let pageNum;
                                if (totalPages <= 5) {
                                  pageNum = i + 1;
                                } else if (currentPage <= 3) {
                                  pageNum = i + 1;
                                } else if (currentPage >= totalPages - 2) {
                                  pageNum = totalPages - 4 + i;
                                } else {
                                  pageNum = currentPage - 2 + i;
                                }
                                
                                return (
                                  <button
                                    key={pageNum}
                                    onClick={() => setCurrentPage(pageNum)}
                                    className={`w-10 h-10 rounded-lg font-semibold transition-colors ${
                                      currentPage === pageNum
                                        ? 'bg-orange-500 text-white'
                                        : 'border border-gray-300 text-gray-700 hover:bg-gray-100'
                                    }`}
                                  >
                                    {pageNum}
                                  </button>
                                );
                              })}
                              
                              {totalPages > 5 && currentPage < totalPages - 2 && (
                                <>
                                  <span className="px-2 text-gray-500">...</span>
                                  <button
                                    onClick={() => setCurrentPage(totalPages)}
                                    className={`w-10 h-10 rounded-lg font-semibold border border-gray-300 text-gray-700 hover:bg-gray-100 transition-colors ${
                                      currentPage === totalPages ? 'bg-orange-500 text-white' : ''
                                    }`}
                                  >
                                    {totalPages}
                                  </button>
                                </>
                              )}
                            </div>
                            
                            <button
                              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                              disabled={currentPage === totalPages}
                              className={`p-2 rounded-lg border transition-colors ${
                                currentPage === totalPages
                                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                                  : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                              }`}
                            >
                              <ChevronRight className="w-5 h-5" />
                            </button>
                          </div>
                          
                          <div className="text-sm text-gray-600 mt-4 sm:mt-0">
                            Page {currentPage} of {totalPages}
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                )}

         {/* Subjects View */}
         {selectedView === 'subjects' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
              {subjectPerformance.map((subject) => (
                <div key={subject.subject} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-bold text-gray-900">{subject.subject}</h4>
                    <BookOpen className="w-5 h-5 text-orange-500" />
                  </div>
                  <div className="text-3xl font-bold text-orange-600 mb-2">{subject.avgScore}</div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-green-600">A: {subject.A ? subject.A.toFixed(1) : 0}%</span>
                      <span className="text-blue-600">B: {subject.B ? subject.B.toFixed(1) : 0}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-yellow-600">C: {subject.C ? subject.C.toFixed(1) : 0}%</span>
                      <span className="text-red-600">D: {subject.D ? subject.D.toFixed(1) : 0}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Subject-wise Grade Distribution</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={subjectPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="subject" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="A" stackId="a" fill="#10b981" name="A Grade %" />
                  <Bar dataKey="B" stackId="a" fill="#3b82f6" name="B Grade %" />
                  <Bar dataKey="C" stackId="a" fill="#f59e0b" name="C Grade %" />
                  <Bar dataKey="D" stackId="a" fill="#ef4444" name="D Grade %" />
                  <Bar dataKey="F" stackId="a" fill="#991b1b" name="F Grade %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Subject Performance Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={subjectPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="subject" />
                  <YAxis domain={[60, 85]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="avgScore" stroke="#f97316" strokeWidth={3} name="Average Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default EducationDashboard
