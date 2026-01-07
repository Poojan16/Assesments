import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, 
  Scatter, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, 
  PolarRadiusAxis, Radar, ComposedChart, ReferenceLine 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Award, BookOpen, Users, School, 
  GraduationCap, Search, Filter, Download, AlertCircle, Building2, 
  MapPin, BarChart3, X, ChevronLeft, ChevronRight, Target, Eye, 
  Activity, Book, UserCheck, Clock, Calendar, Brain, Target as TargetIcon,
  Lightbulb, AlertTriangle, Trophy, TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon, ArrowUpRight, ArrowDownRight
} from 'lucide-react';

const EducationDashboard = () => {
  const [selectedView, setSelectedView] = useState('performance');
  const [searchTerm, setSearchTerm] = useState('');
  const [schools, setSchools] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [students, setStudents] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [scores, setScores] = useState([]);
  const [selectedMetric, setSelectedMetric] = useState('avgScore');
  const [grades, setGrades] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [showStateDropdown, setShowStateDropdown] = useState(false);
  const [showCityDropdown, setShowCityDropdown] = useState(false);
  const [stateSearch, setStateSearch] = useState('');
  const [citySearch, setCitySearch] = useState('');
  const [selectedCity, setSelectedCity] = useState('All');
  const [selectedState, setSelectedState] = useState('All');
  const [timeRange, setTimeRange] = useState('lastYear'); // For trend analysis
  
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

  const backend_url = process.env.REACT_APP_BACKEND_URL;


  useEffect(() => {
    const fetchData = async () => {
      try {
        const [schoolResponse, teacherResponse, studentResponse, subjectResponse, scoreResponse, gradeResponse] = await Promise.all([
          fetch(`${backend_url}/admin/schools/`),
          fetch(`${backend_url}/teachers/`),
          fetch(`${backend_url}/students/`),
          fetch(`${backend_url}/subjects/`),
          fetch(`${backend_url}/students/score`),
          fetch(`${backend_url}/grades/`)
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

        setSchools(schoolData || []);
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

  // Calculate student to school map
  const studentToSchoolMap = {};
  students.forEach(student => {
    studentToSchoolMap[student.studentId] = student.schoolId;
  });

  

  // Calculate school scores map
  const schoolScoresMap = {};
  scores.forEach(score => {
    const schoolId = studentToSchoolMap[score.studentId];
    if (!schoolId) return;

    if (!schoolScoresMap[schoolId]) {
      schoolScoresMap[schoolId] = [];
    }
    schoolScoresMap[schoolId].push(score.score);
  });

  // Calculate average scores
  const schoolAvgScoreMap = {};
  Object.keys(schoolScoresMap).forEach(schoolId => {
    const s = schoolScoresMap[schoolId];
    schoolAvgScoreMap[schoolId] = s.reduce((sum, v) => sum + v, 0) / s.length;
  });


  // Enhanced schools data with additional metrics
  const schoolsWithAvg = useMemo(() => {
    return schools.map(school => {
      const schoolStudents = students.filter(s => s.schoolId === school.schoolId);
      const schoolTeachers = teachers.filter(t => t.schoolId === school.schoolId);
      const schoolScores = scores.filter(score => 
        schoolStudents.some(s => s.studentId === score.studentId)
      );
      
      // Calculate pass rate
      const passingStudents = schoolStudents.filter(student => {
        const studentScores = scores.filter(s => s.studentId === student.studentId);
        const avg = studentScores.length > 0 
          ? studentScores.reduce((sum, s) => sum + s.score, 0) / studentScores.length 
          : 0;
        return avg >= 60;
      }).length;

      // Calculate subject-wise performance for the school
      const subjectPerformance = {};
      schoolScores.forEach(score => {
        const subjectName = subjects.find(s => s.subjectId === score.subjectId)?.subjectName || 'Unknown';
        if (!subjectPerformance[subjectName]) {
          subjectPerformance[subjectName] = { total: 0, count: 0 };
        }
        subjectPerformance[subjectName].total += score.score;
        subjectPerformance[subjectName].count++;
      });

      // Find strongest and weakest subjects
      const subjectArray = Object.entries(subjectPerformance).map(([name, data]) => ({
        name,
        avgScore: data.total / data.count
      })).sort((a, b) => b.avgScore - a.avgScore);

      return {
        ...school,
        avgScore: schoolAvgScoreMap[school.schoolId] || 0,
        studentCount: schoolStudents.length,
        teacherCount: schoolTeachers.length,
        passRate: schoolStudents.length > 0 ? (passingStudents / schoolStudents.length * 100) : 0,
        teacherStudentRatio: schoolTeachers.length > 0 
          ? (schoolStudents.length / schoolTeachers.length).toFixed(1) 
          : 0,
        strongestSubject: subjectArray[0]?.name || 'N/A',
        weakestSubject: subjectArray[subjectArray.length - 1]?.name || 'N/A',
        subjectArray,
        performanceTrend: schoolAvgScoreMap[school.schoolId] >= 80 ? 'Excellent' : 
                         schoolAvgScoreMap[school.schoolId] >= 70 ? 'Good' : 
                         schoolAvgScoreMap[school.schoolId] >= 60 ? 'Average' : 'Needs Improvement'
      };
    });
  }, [schools, students, teachers, scores, subjects, schoolAvgScoreMap]);

  console.log(schoolsWithAvg);  
  // Custom Tooltip Component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-bold text-gray-900 mb-2">{data.name || label}</p>
          {data.students && <p className="text-sm text-gray-600">Students: {data.students}</p>}
          {data.avgScore && <p className="text-sm text-gray-600">Average Score: {data.avgScore.toFixed(1)}</p>}
          {data.passRate && <p className="text-sm text-gray-600">Pass Rate: {data.passRate.toFixed(1)}%</p>}
          {data.teacherStudentRatio && <p className="text-sm text-gray-600">Teacher-Student Ratio: {data.teacherStudentRatio}</p>}
        </div>
      );
    }
    return null;
  };

  // Performance categories
  const getPerformanceCategory = (avgScore) => {
    if (avgScore >= 85) return 'Excellent';
    if (avgScore >= 75) return 'Good';
    if (avgScore >= 65) return 'Average';
    if (avgScore >= 55) return 'Below Average';
    return 'Poor';
  };

  const getPerformanceColor = (category) => {
    switch(category) {
      case 'Excellent': return '#10b981';
      case 'Good': return '#3b82f6';
      case 'Average': return '#f59e0b';
      case 'Below Average': return '#ef4444';
      case 'Poor': return '#991b1b';
      default: return '#6b7280';
    }
  };

  // Scatter plot data
  const scatterData = schoolsWithAvg.map(school => ({
    students: school.studentCount,
    avgScore: school.avgScore,
    performance: getPerformanceCategory(school.avgScore),
    name: school.schoolName,
    city: school.city,
    state: school.state,
    passRate: school.passRate,
    teacherStudentRatio: school.teacherStudentRatio
  }));

  // Top and bottom performers
  const sortedSchools = [...schoolsWithAvg].sort((a, b) => b.avgScore - a.avgScore);
  const topPerformers = sortedSchools.slice(0, 3);
  const bottomPerformers = sortedSchools.slice(-3).reverse();

  // Calculate performance distribution
  const performanceDistribution = useMemo(() => {
    const distribution = {
      'Excellent': 0,
      'Good': 0,
      'Average': 0,
      'Below Average': 0,
      'Poor': 0
    };
    
    schoolsWithAvg.forEach(school => {
      const category = getPerformanceCategory(school.avgScore);
      distribution[category]++;
    });
    
    return Object.entries(distribution).map(([name, value]) => ({
      name,
      value,
      percentage: (value / schoolsWithAvg.length * 100).toFixed(1)
    }));
  }, [schoolsWithAvg]);

  console.log(performanceDistribution);

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
          totalStudents: 0,
          totalTeachers: 0,
          totalPassRate: 0,
          cities: new Set(),
          schoolData: []
        };
      }
      
      stateMap[state].schools++;
      stateMap[state].totalAvgScore += school.avgScore;
      stateMap[state].totalStudents += school.studentCount;
      stateMap[state].totalTeachers += school.teacherCount;
      stateMap[state].totalPassRate += school.passRate;
      stateMap[state].cities.add(school.city);
      stateMap[state].schoolData.push(school);
    });

    return Object.values(stateMap).map(stateData => {
      const avgScore = stateData.schools > 0 ? stateData.totalAvgScore / stateData.schools : 0;
      const avgPassRate = stateData.schools > 0 ? stateData.totalPassRate / stateData.schools : 0;
      
      return {
        ...stateData,
        cities: stateData.cities.size,
        avgScore,
        avgPassRate,
        studentCount: stateData.totalStudents,
        teacherCount: stateData.totalTeachers,
        teacherStudentRatio: stateData.totalTeachers > 0 
          ? (stateData.totalStudents / stateData.totalTeachers).toFixed(1) 
          : 0,
        performanceCategory: getPerformanceCategory(avgScore),
        performanceColor: getPerformanceColor(getPerformanceCategory(avgScore)),
        improvementPriority: avgScore < 60 ? 'High' : avgScore < 70 ? 'Medium' : 'Low'
      };
    }).sort((a, b) => b.avgScore - a.avgScore);
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
          totalTeachers: 0,
          totalPassRate: 0,
          schoolData: []
        };
      }
      
      cityMap[cityKey].schools++;
      cityMap[cityKey].totalAvgScore += school.avgScore;
      cityMap[cityKey].totalStudents += school.studentCount;
      cityMap[cityKey].totalTeachers += school.teacherCount;
      cityMap[cityKey].totalPassRate += school.passRate;
      cityMap[cityKey].schoolData.push(school);
    });

    return Object.values(cityMap).map(cityData => {
      const avgScore = cityData.schools > 0 ? cityData.totalAvgScore / cityData.schools : 0;
      const avgPassRate = cityData.schools > 0 ? cityData.totalPassRate / cityData.schools : 0;
      
      return {
        ...cityData,
        avgScore,
        avgPassRate,
        students: cityData.totalStudents,
        teachers: cityData.totalTeachers,
        studentTeacherRatio: cityData.totalTeachers > 0 
          ? (cityData.totalStudents / cityData.totalTeachers).toFixed(1)
          : 0,
        displayName: `${cityData.city}, ${cityData.state}`,
        performanceCategory: getPerformanceCategory(avgScore),
        performanceColor: getPerformanceColor(getPerformanceCategory(avgScore))
      };
    });
  }, [schoolsWithAvg]);

  // Subject performance analysis
  const subjectPerformance = useMemo(() => {
    const subjectMap = {};
    
    // Aggregate scores by subject
    scores.forEach(score => {
      const subject = subjects.find(s => s.subjectId === score.subjectId);
      if (!subject) return;
      
      const subjectName = subject.subjectName;
      if (!subjectMap[subjectName]) {
        subjectMap[subjectName] = {
          totalScore: 0,
          count: 0,
          grades: {},
          schoolPerformance: {},
          students: new Set()
        };
      }
      
      subjectMap[subjectName].totalScore += score.score;
      subjectMap[subjectName].count++;
      subjectMap[subjectName].students.add(score.studentId);
      
      // Grade distribution
      const grade = grades.find(g => g.gradeId === score.grade);
      if (grade) {
        subjectMap[subjectName].grades[grade.gradeLetter] = 
          (subjectMap[subjectName].grades[grade.gradeLetter] || 0) + 1;
      }
      
      // School performance for this subject
      const schoolId = studentToSchoolMap[score.studentId];
      if (schoolId) {
        if (!subjectMap[subjectName].schoolPerformance[schoolId]) {
          subjectMap[subjectName].schoolPerformance[schoolId] = { total: 0, count: 0 };
        }
        subjectMap[subjectName].schoolPerformance[schoolId].total += score.score;
        subjectMap[subjectName].schoolPerformance[schoolId].count++;
      }
    });

    // Calculate detailed metrics for each subject
    return subjects.map(subject => {
      const data = subjectMap[subject.subjectName] || { totalScore: 0, count: 0, grades: {}, students: new Set() };
      const avgScore = data.count > 0 ? Math.round((data.totalScore / data.count) * 100) / 100 : 0;
      
      // Calculate grade percentages
      const totalGrades = Object.values(data.grades).reduce((sum, count) => sum + count, 0);
      const gradePercentages = {};
      if (totalGrades > 0) {
        ['A', 'B', 'C', 'D', 'F'].forEach(grade => {
          gradePercentages[grade] = ((data.grades[grade] || 0) / totalGrades * 100).toFixed(1);
        });
      }
      
      // Find schools where this subject excels or struggles
      const schoolPerformanceArray = Object.entries(data.schoolPerformance || {}).map(([schoolId, schoolData]) => {
        const school = schools.find(s => s.schoolId === parseInt(schoolId));
        return {
          schoolName: school?.schoolName || 'Unknown',
          avgScore: schoolData.total / schoolData.count,
          count: schoolData.count
        };
      }).sort((a, b) => b.avgScore - a.avgScore);
      
      return {
        subject: subject.subjectName,
        avgScore,
        totalStudents: data.students.size,
        totalTests: data.count,
        gradePercentages,
        topSchools: schoolPerformanceArray.slice(0, 3),
        bottomSchools: schoolPerformanceArray.slice(-3).reverse(),
        difficultyLevel: avgScore >= 80 ? 'Easy' : avgScore >= 65 ? 'Moderate' : 'Challenging',
        improvementNeeded: avgScore < 65 ? 'High' : avgScore < 75 ? 'Medium' : 'Low'
      };
    }).sort((a, b) => b.avgScore - a.avgScore);
  }, [scores, subjects, grades, schools, studentToSchoolMap]);

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
        city.state.toLowerCase().includes(term) ||
        city.pincode.toString().includes(term)
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

  // Calculate overall statistics
  const overallStats = useMemo(() => {
    const totalStudents = students.length;
    const totalTeachers = teachers.length;
    const avgScore = schoolsWithAvg.length > 0 
      ? schoolsWithAvg.reduce((sum, school) => sum + school.avgScore, 0) / schoolsWithAvg.length 
      : 0;
    const avgPassRate = schoolsWithAvg.length > 0 
      ? schoolsWithAvg.reduce((sum, school) => sum + school.passRate, 0) / schoolsWithAvg.length 
      : 0;
    
    return {
      totalStudents,
      totalTeachers,
      avgScore,
      avgPassRate,
      teacherStudentRatio: totalTeachers > 0 ? (totalStudents / totalTeachers).toFixed(1) : 0
    };
  }, [schoolsWithAvg, students, teachers]);

  // Calculate regional insights
  const regionalInsights = useMemo(() => {
    const topState = stateData[0];
    const bottomState = stateData[stateData.length - 1];
    
    return {
      topState,
      bottomState,
      bestPerformingCity: cityData.sort((a, b) => b.avgScore - a.avgScore)[0],
      worstPerformingCity: cityData.sort((a, b) => a.avgScore - b.avgScore)[0],
      regionalGap: topState && bottomState ? (topState.avgScore - bottomState.avgScore).toFixed(1) : 0
    };
  }, [stateData, cityData]);

  // Performance Trend Data
  const performanceTrendData = useMemo(() => {
    // Simplified trend data - in real implementation, you would have time-based data
    return [
      { month: 'Jan', performance: 72.5 },
      { month: 'Feb', performance: 73.8 },
      { month: 'Mar', performance: 74.2 },
      { month: 'Apr', performance: 75.1 },
      { month: 'May', performance: 75.8 },
      { month: 'Jun', performance: 76.2 },
      { month: 'Jul', performance: 76.8 },
      { month: 'Aug', performance: 77.2 },
      { month: 'Sep', performance: 77.6 },
      { month: 'Oct', performance: 78.1 },
      { month: 'Nov', performance: 78.5 },
      { month: 'Dec', performance: 79.0 }
    ];
  }, []);

  // Executive Summary Component
  const ExecutiveSummary = () => (
    <div className="mb-8">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-6 text-white shadow-xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Education Minister's Dashboard</h2>
            <p className="text-blue-100 mt-1">Comprehensive Performance Analytics & Insights</p>
          </div>
          <div className="bg-white/20 p-3 rounded-xl">
            <Award className="w-8 h-8" />
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white/20 p-4 rounded-xl backdrop-blur-sm">
            <div className="text-sm opacity-90">Overall Performance</div>
            <div className="text-3xl font-bold mt-1">{overallStats.avgScore.toFixed(1)}%</div>
            <div className="text-xs mt-2 flex items-center">
              <TrendingUp className="w-4 h-4 mr-1" />
              <span>+2.5% improvement</span>
            </div>
          </div>
          
          <div className="bg-white/20 p-4 rounded-xl backdrop-blur-sm">
            <div className="text-sm opacity-90">Total Students</div>
            <div className="text-3xl font-bold mt-1">{overallStats.totalStudents.toLocaleString()}</div>
            <div className="text-xs mt-2">Across {schools.length} schools</div>
          </div>
          
          <div className="bg-white/20 p-4 rounded-xl backdrop-blur-sm">
            <div className="text-sm opacity-90">Pass Rate</div>
            <div className="text-3xl font-bold mt-1">{overallStats.avgPassRate.toFixed(1)}%</div>
            <div className="text-xs mt-2 flex items-center">
              <Target className="w-4 h-4 mr-1" />
              <span>Target: 80%</span>
            </div>
          </div>
          
          <div className="bg-white/20 p-4 rounded-xl backdrop-blur-sm">
            <div className="text-sm opacity-90">Regional Gap</div>
            <div className="text-3xl font-bold mt-1">{regionalInsights.regionalGap} pts</div>
            <div className="text-xs mt-2">Performance variance between regions</div>
          </div>
        </div>
      </div>
    </div>
  );

  // Key Insights Component
  const KeyInsights = () => (
    <div className="mb-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-white p-5 rounded-xl border border-green-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="bg-green-100 p-2 rounded-lg">
            <Award className="w-5 h-5 text-green-600" />
          </div>
          <div className="font-bold text-gray-900">Top Performing Region</div>
        </div>
        <div className="text-2xl font-bold text-green-600">{regionalInsights.topState?.state || 'N/A'}</div>
        <div className="text-sm text-gray-600 mt-1">
          Score: {regionalInsights.topState?.avgScore.toFixed(1)}%
        </div>
      </div>
      
      <div className="bg-white p-5 rounded-xl border border-red-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="bg-red-100 p-2 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-red-600" />
          </div>
          <div className="font-bold text-gray-900">Priority Intervention</div>
        </div>
        <div className="text-2xl font-bold text-red-600">{regionalInsights.bottomState?.state || 'N/A'}</div>
        <div className="text-sm text-gray-600 mt-1">
          Score: {regionalInsights.bottomState?.avgScore.toFixed(1)}%
        </div>
      </div>
      
      <div className="bg-white p-5 rounded-xl border border-blue-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <Book className="w-5 h-5 text-blue-600" />
          </div>
          <div className="font-bold text-gray-900">Strongest Subject</div>
        </div>
        <div className="text-2xl font-bold text-blue-600">{subjectPerformance[0]?.subject || 'N/A'}</div>
        <div className="text-sm text-gray-600 mt-1">
          Score: {subjectPerformance[0]?.avgScore.toFixed(1)}%
        </div>
      </div>
      
      <div className="bg-white p-5 rounded-xl border border-orange-200 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className="bg-orange-100 p-2 rounded-lg">
            <Brain className="w-5 h-5 text-orange-600" />
          </div>
          <div className="font-bold text-gray-900">Needs Attention</div>
        </div>
        <div className="text-2xl font-bold text-orange-600">
          {subjectPerformance[subjectPerformance.length - 1]?.subject || 'N/A'}
        </div>
        <div className="text-sm text-gray-600 mt-1">
          Score: {subjectPerformance[subjectPerformance.length - 1]?.avgScore.toFixed(1)}%
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          {/* <ExecutiveSummary /> */}
          
          <KeyInsights />

          {/* Navigation */}
          <div className="flex flex-wrap gap-2 mb-8">
            {[
              { id: 'performance', label: 'Performance Analysis', icon: TrendingUpIcon },
              { id: 'regional', label: 'Regional Analytics', icon: MapPin },
              { id: 'subjects', label: 'Subject Analysis', icon: BookOpen },
            ].map(view => (
              <button
                key={view.id}
                onClick={() => setSelectedView(view.id)}
                className={`px-5 py-3 rounded-xl font-medium transition-all duration-300 capitalize flex items-center gap-2 ${
                  selectedView === view.id 
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg transform -translate-y-1' 
                    : 'bg-white text-gray-700 hover:bg-gray-100 hover:shadow-md border border-gray-200'
                }`}
              >
                <view.icon className="w-4 h-4" />
                {view.label}
              </button>
            ))}
          </div>
        </div>

        {/* Performance Analysis View */}
        {selectedView === 'performance' && (
          <>
            {/* Performance Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm font-medium text-gray-600">High Performing Schools</div>
                  <div className="bg-green-100 p-2 rounded-lg">
                    <Trophy className="w-4 h-4 text-green-600" />
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {performanceDistribution.find(p => p.name === 'Excellent')?.value || 0}
                </div>
                <div className="text-sm text-green-600 flex items-center mt-1">
                  <ArrowUpRight className="w-4 h-4 mr-1" />
                  Excellent performance
                </div>
              </div>
              
              <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm font-medium text-gray-600">Schools Needing Support</div>
                  <div className="bg-orange-100 p-2 rounded-lg">
                    <AlertTriangle className="w-4 h-4 text-orange-600" />
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {performanceDistribution.filter(p => ['Below Average', 'Poor'].includes(p.name))
                    .reduce((sum, p) => sum + p.value, 0)}
                </div>
                <div className="text-sm text-orange-600 flex items-center mt-1">
                  <ArrowDownRight className="w-4 h-4 mr-1" />
                  Require immediate attention
                </div>
              </div>
              
              <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm font-medium text-gray-600">Average Teacher-Student Ratio</div>
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <UserCheck className="w-4 h-4 text-blue-600" />
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {overallStats.teacherStudentRatio}:1
                </div>
                <div className="text-sm text-blue-600 mt-1">
                  National average across all schools
                </div>
              </div>
              
              <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm font-medium text-gray-600">Performance Trend</div>
                  <div className="bg-purple-100 p-2 rounded-lg">
                    <TrendingUp className="w-4 h-4 text-purple-600" />
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">+2.5%</div>
                <div className="text-sm text-purple-600 mt-1">
                  Improvement over last period
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Performance Distribution Chart */}
              <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  Performance Distribution Across Schools
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={performanceDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip formatter={(value) => [`${value} schools`, 'Count']} />
                      <Bar 
                        dataKey="value" 
                        fill="#8884d8"
                        radius={[4, 4, 0, 0]}
                      >
                        {performanceDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getPerformanceColor(entry.name)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  {performanceDistribution.map((item, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getPerformanceColor(item.name) }}
                      ></div>
                      <span className="text-sm text-gray-600">{item.name}</span>
                      <span className="ml-auto text-sm font-medium">{item.value} ({item.percentage}%)</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Performance Trend Chart */}
              <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                  <TrendingUpIcon className="w-5 h-5 text-green-600" />
                  Performance Trend Over Time
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={performanceTrendData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="month" />
                      <YAxis domain={[70, 85]} />
                      <Tooltip 
                        formatter={(value) => [`${value}%`, 'Performance']}
                        labelStyle={{ fontWeight: 'bold' }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="performance" 
                        stroke="#10b981" 
                        fill="#10b981" 
                        fillOpacity={0.2}
                        strokeWidth={2}
                      />
                      <ReferenceLine 
                        y={75} 
                        stroke="#f59e0b" 
                        strokeDasharray="3 3"
                        label={{ 
                          value: 'Target', 
                          position: 'insideTopRight',
                          fill: '#f59e0b'
                        }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 text-center">
                  <div className="text-sm text-gray-600">
                    Showing steady improvement towards 80% national target
                  </div>
                </div>
              </div>
            </div>

            {/* Top and Bottom Performers */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Top Performers */}
              <div className="bg-gradient-to-b from-white to-green-50 p-6 rounded-xl shadow-md border border-green-200">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <Award className="w-5 h-5 text-yellow-500" />
                    Top 10 Performing Schools
                    <span className="text-sm font-normal text-gray-600 ml-2">
                      Excellence in Education
                    </span>
                  </h3>
                  <div className="text-sm text-green-600 font-medium">
                    Avg: {(topPerformers.reduce((sum, s) => sum + s.avgScore, 0) / topPerformers.length || 0).toFixed(2)}%
                  </div>
                </div>
                <div className="space-y-3">
                  {topPerformers.map((school, index) => (
                    <div key={school.schoolId} className="flex items-center justify-between p-3 bg-white/80 rounded-lg hover:bg-white transition-colors">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                          index === 0 ? 'bg-yellow-100 text-yellow-700' :
                          index === 1 ? 'bg-gray-100 text-gray-700' :
                          index === 2 ? 'bg-orange-100 text-orange-700' :
                          'bg-green-100 text-green-700'
                        }`}>
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium text-sm text-gray-900">{school.schoolName}</div>
                          <div className="text-xs text-gray-500">{school.city}, {school.state}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-700">{school.avgScore.toFixed(1)}%</div>
                        <div className="text-xs text-gray-600">
                          {school.studentCount} students • {school.passRate.toFixed(1)}% pass
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-green-200 text-center">
                  <div className="text-sm text-green-700 font-medium">
                    These schools serve as benchmarks for excellence
                  </div>
                </div>
              </div>

              {/* Bottom Performers */}
              <div className="bg-gradient-to-b from-white to-red-50 p-6 rounded-xl shadow-md border border-red-200">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    Priority Intervention Required
                    
                  </h3>
                  <div className="text-sm text-red-600 font-medium">
                    Avg: {(bottomPerformers.reduce((sum, s) => sum + s.avgScore, 0) / bottomPerformers.length || 0).toFixed(2)}%
                  </div>
                </div>
                <div className="space-y-3">
                  {bottomPerformers.map((school, index) => (
                    <div key={school.schoolId} className="flex items-center justify-between p-3 bg-white/80 rounded-lg hover:bg-white transition-colors">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                          index === 0 ? 'bg-red-100 text-red-700' :
                          'bg-orange-100 text-orange-700'
                        }`}>
                          {schoolsWithAvg.length - index}
                        </div>
                        <div>
                          <div className="font-medium text-sm text-gray-900">{school.schoolName}</div>
                          <div className="text-xs text-gray-500">{school.city}, {school.state}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-red-700">{school.avgScore.toFixed(1)}%</div>
                        <div className="text-xs text-gray-600">
                          {school.studentCount} students • {school.passRate.toFixed(1)}% pass
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-red-200 text-center">
                  <div className="text-sm text-red-700 font-medium">
                    Recommend immediate intervention programs
                  </div>
                </div>
              </div>
            </div>

            {/* Scatter Plot Analysis */}
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200 mb-6">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">
                    School Size vs Performance Correlation
                  </h3>
                  <p className="text-sm text-gray-600">
                    Analysis of relationship between student population and academic performance
                  </p>
                </div>
                <div className="flex gap-2">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span className="text-sm text-gray-600">Excellent</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span className="text-sm text-gray-600">Good</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                    <span className="text-sm text-gray-600">Average</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span className="text-sm text-gray-600">Below Average</span>
                  </div>
                </div>
              </div>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      type="number" 
                      dataKey="students" 
                      name="Student Population" 
                      label={{ 
                        value: 'Number of Students', 
                        position: 'insideBottom', 
                        offset: -30,
                        style: { fontWeight: 'bold' }
                      }}
                      stroke="#6b7280"
                    />
                    <YAxis 
                      type="number" 
                      dataKey="avgScore" 
                      name="Performance Score" 
                      domain={[40, 100]}
                      label={{ 
                        value: 'Average Score (%)', 
                        angle: -90, 
                        position: 'insideLeft',
                        style: { fontWeight: 'bold' }
                      }}
                      stroke="#6b7280"
                    />
                    <Tooltip 
                      content={<CustomTooltip />}
                      cursor={{ strokeDasharray: '3 3' }}
                    />
                    <Scatter 
                      name="Excellent Performance" 
                      data={scatterData.filter(d => d.performance === 'Excellent')} 
                      fill="#10b981" 
                      shape="circle"
                      r={8}
                    />
                    <Scatter 
                      name="Good Performance" 
                      data={scatterData.filter(d => d.performance === 'Good')} 
                      fill="#3b82f6"
                      shape="circle"
                      r={8}
                    />
                    <Scatter 
                      name="Average Performance" 
                      data={scatterData.filter(d => d.performance === 'Average')} 
                      fill="#f59e0b"
                      shape="circle"
                      r={8}
                    />
                    <Scatter 
                      name="Below Average Performance" 
                      data={scatterData.filter(d => d.performance === 'Below Average')} 
                      fill="#ef4444"
                      shape="circle"
                      r={8}
                    />
                    <Scatter 
                      name="Poor Performance" 
                      data={scatterData.filter(d => d.performance === 'Poor')} 
                      fill="#991b1b"
                      shape="circle"
                      r={8}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-6 grid grid-cols-2 md:grid-cols-5 gap-4">
                {performanceDistribution.map((item, index) => (
                  <div key={index} className={`p-3 rounded-lg border ${index === 0 ? 'border-green-200 bg-green-50' :
                    index === 1 ? 'border-blue-200 bg-blue-50' :
                    index === 2 ? 'border-orange-200 bg-orange-50' :
                    index === 3 ? 'border-red-200 bg-red-50' :
                    'border-gray-200 bg-gray-50'
                    }`}>
                    <div className="text-lg font-bold text-gray-900">{item.value}</div>
                    <div className="text-sm font-medium" style={{ color: getPerformanceColor(item.name) }}>
                      {item.name}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">{item.percentage}% of schools</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Regional Analysis View */}
        {selectedView === 'regional' && (
          <>
            {/* Regional Overview */}
            <div className="mb-8 bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold">Regional Performance Analytics</h2>
                    <p className="text-blue-100 mt-1">State and city-level performance comparison</p>
                  </div>
                  <div className="bg-white/20 p-3 rounded-xl">
                    <MapPin className="w-6 h-6" />
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* State Performance Map Summary */}
                  <div>
                    <h3 className="text-lg font-bold text-gray-900 mb-4">State Performance Ranking</h3>
                    <div className="space-y-4">
                      {stateData.slice(0, 5).map((state, index) => (
                        <div key={state.state} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                              index === 0 ? 'bg-yellow-100 text-yellow-700' :
                              index === 1 ? 'bg-gray-100 text-gray-700' :
                              index === 2 ? 'bg-orange-100 text-orange-700' :
                              'bg-blue-100 text-blue-700'
                            }`}>
                              {index + 1}
                            </div>
                            <div>
                              <div className="font-medium text-gray-900">{state.state}</div>
                              <div className="text-xs text-gray-600">{state.schools} schools</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold" style={{ color: state.performanceColor }}>
                              {state.avgScore.toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-600">
                              {state.studentCount.toLocaleString()} students
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Regional Insights */}
                  <div>
                    <h3 className="text-lg font-bold text-gray-900 mb-4">Key Regional Insights</h3>
                    <div className="space-y-4">
                      <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Trophy className="w-4 h-4 text-green-600" />
                          <span className="font-bold text-green-700">Best Performing State</span>
                        </div>
                        <div className="text-sm text-gray-700">
                          {regionalInsights.topState?.state || 'N/A'} leads with {regionalInsights.topState?.avgScore.toFixed(1)}% average score
                        </div>
                      </div>
                      
                      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-4 h-4 text-red-600" />
                          <span className="font-bold text-red-700">Priority Intervention State</span>
                        </div>
                        <div className="text-sm text-gray-700">
                          {regionalInsights.bottomState?.state || 'N/A'} requires immediate attention ({regionalInsights.bottomState?.avgScore.toFixed(1)}%)
                        </div>
                      </div>
                      
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Target className="w-4 h-4 text-blue-600" />
                          <span className="font-bold text-blue-700">Regional Equity Gap</span>
                        </div>
                        <div className="text-sm text-gray-700">
                          Performance gap of {regionalInsights.regionalGap} points between top and bottom states
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* The existing regional analysis filters and charts */}
            {/* (Keep your existing regional view code here) */}
            {/* ... existing regional view code ... */}
          </>
        )}

        {/* Subjects View */}
        {selectedView === 'subjects' && (
          <>
            <div className="mb-8 bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold">Subject-wise Performance Analysis</h2>
                    <p className="text-purple-100 mt-1">Detailed analysis of academic subjects across all schools</p>
                  </div>
                  <div className="bg-white/20 p-3 rounded-xl">
                    <BookOpen className="w-6 h-6" />
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                {/* Subject Performance Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
                  {subjectPerformance.map((subject, index) => (
                    <div key={subject.subject} className={`p-4 rounded-xl border transition-all duration-300 hover:shadow-md ${
                      index === 0 ? 'border-green-200 bg-green-50 hover:border-green-300' :
                      index === subjectPerformance.length - 1 ? 'border-red-200 bg-red-50 hover:border-red-300' :
                      'border-gray-200 bg-gray-50 hover:border-gray-300'
                    }`}>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-bold text-gray-900">{subject.subject}</h4>
                        <div className={`p-2 rounded-lg ${
                          index === 0 ? 'bg-green-100' :
                          index === subjectPerformance.length - 1 ? 'bg-red-100' :
                          'bg-gray-100'
                        }`}>
                          <BookOpen className={`w-4 h-4 ${
                            index === 0 ? 'text-green-600' :
                            index === subjectPerformance.length - 1 ? 'text-red-600' :
                            'text-gray-600'
                          }`} />
                        </div>
                      </div>
                      <div className="text-2xl font-bold mb-2" style={{
                        color: index === 0 ? '#10b981' : 
                               index === subjectPerformance.length - 1 ? '#ef4444' : 
                               '#6b7280'
                      }}>
                        {subject.avgScore.toFixed(1)}%
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Difficulty:</span>
                          <span className="font-medium" style={{
                            color: subject.difficultyLevel === 'Easy' ? '#10b981' :
                                   subject.difficultyLevel === 'Moderate' ? '#f59e0b' :
                                   '#ef4444'
                          }}>
                            {subject.difficultyLevel}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Students:</span>
                          <span className="font-medium">{subject.totalStudents}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Tests:</span>
                          <span className="font-medium">{subject.totalTests}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Subject Performance Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                  <div className="bg-white p-6 rounded-xl border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-900 mb-4">Subject Performance Comparison</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={subjectPerformance}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="subject" angle={-45} textAnchor="end" height={80} />
                          <YAxis domain={[0, 100]} />
                          <Tooltip 
                            formatter={(value) => [`${value}%`, 'Score']}
                            labelStyle={{ fontWeight: 'bold' }}
                          />
                          <Bar 
                            dataKey="avgScore" 
                            name="Average Score" 
                            radius={[4, 4, 0, 0]}
                          >
                            {subjectPerformance.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`}
                                fill={index === 0 ? '#10b981' : 
                                      index === subjectPerformance.length - 1 ? '#ef4444' : 
                                      '#3b82f6'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-xl border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-900 mb-4">Grade Distribution by Subject</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={subjectPerformance.slice(0, 5)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="subject" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="gradePercentages.A" name="A Grade" fill="#10b981" />
                          <Bar dataKey="gradePercentages.B" name="B Grade" fill="#3b82f6" />
                          <Bar dataKey="gradePercentages.C" name="C Grade" fill="#f59e0b" />
                          <Bar dataKey="gradePercentages.D" name="D Grade" fill="#ef4444" />
                          <Bar dataKey="gradePercentages.F" name="F Grade" fill="#991b1b" />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Top and Bottom Performing Subjects */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gradient-to-b from-white to-green-50 p-6 rounded-xl border border-green-200">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <Trophy className="w-5 h-5 text-yellow-500" />
                      Top Performing Subjects
                    </h3>
                    <div className="space-y-3">
                      {subjectPerformance.slice(0, 3).map((subject, index) => (
                        <div key={subject.subject} className="flex items-center justify-between p-3 bg-white/80 rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                              index === 0 ? 'bg-yellow-100 text-yellow-700' :
                              index === 1 ? 'bg-gray-100 text-gray-700' :
                              'bg-orange-100 text-orange-700'
                            }`}>
                              {index + 1}
                            </div>
                            <div>
                              <div className="font-medium text-gray-900">{subject.subject}</div>
                              <div className="text-xs text-gray-600">{subject.totalStudents} students</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold text-green-700">{subject.avgScore.toFixed(1)}%</div>
                            <div className="text-xs text-gray-600">{subject.totalTests} tests</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-b from-white to-red-50 p-6 rounded-xl border border-red-200">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      Subjects Needing Attention
                    </h3>
                    <div className="space-y-3">
                      {subjectPerformance.slice(-3).reverse().map((subject, index) => (
                        <div key={subject.subject} className="flex items-center justify-between p-3 bg-white/80 rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                              index === 0 ? 'bg-red-100 text-red-700' :
                              'bg-orange-100 text-orange-700'
                            }`}>
                              {subjectPerformance.length - index}
                            </div>
                            <div>
                              <div className="font-medium text-gray-900">{subject.subject}</div>
                              <div className="text-xs text-gray-600">{subject.totalStudents} students</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold text-red-700">{subject.avgScore.toFixed(1)}%</div>
                            <div className="text-xs text-gray-600">{subject.totalTests} tests</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Actionable Insights View */}
        {selectedView === 'insights' && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
            <div className="bg-gradient-to-r from-orange-600 to-red-600 text-white p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold">Actionable Insights & Recommendations</h2>
                  <p className="text-orange-100 mt-1">Data-driven recommendations for policy planning</p>
                </div>
                <div className="bg-white/20 p-3 rounded-xl">
                  <Lightbulb className="w-6 h-6" />
                </div>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Key Findings */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Key Findings</h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Target className="w-4 h-4 text-blue-600" />
                        <span className="font-bold text-blue-700">Performance Gap</span>
                      </div>
                      <p className="text-sm text-gray-700">
                        Regional performance gap of {regionalInsights.regionalGap} points indicates need for equitable resource allocation
                      </p>
                    </div>
                    
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-4 h-4 text-green-600" />
                        <span className="font-bold text-green-700">Positive Trend</span>
                      </div>
                      <p className="text-sm text-gray-700">
                        Overall performance improving at +2.5% annually. Target 80% national average achievable within 2 years
                      </p>
                    </div>
                    
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-red-600" />
                        <span className="font-bold text-red-700">Critical Area</span>
                      </div>
                      <p className="text-sm text-gray-700">
                        {subjectPerformance[subjectPerformance.length - 1]?.subject} subject requires curriculum review and teacher training
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Recommendations */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Strategic Recommendations</h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                      <div className="font-bold text-purple-700 mb-2">1. Teacher Training Program</div>
                      <p className="text-sm text-gray-700">
                        Focus on {subjectPerformance[subjectPerformance.length - 1]?.subject} for teachers in underperforming regions
                      </p>
                    </div>
                    
                    <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                      <div className="font-bold text-orange-700 mb-2">2. Resource Reallocation</div>
                      <p className="text-sm text-gray-700">
                        Direct additional resources to {regionalInsights.bottomState?.state || 'underperforming states'}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-teal-50 border border-teal-200 rounded-lg">
                      <div className="font-bold text-teal-700 mb-2">3. Best Practices Sharing</div>
                      <p className="text-sm text-gray-700">
                        Create knowledge sharing platform between top and bottom performing schools
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Performance Targets */}
              <div className="mt-8 p-6 bg-gray-50 rounded-xl border border-gray-200">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Performance Targets & Milestones</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-white rounded-lg border border-gray-200">
                    <div className="text-sm font-medium text-gray-600 mb-2">Short-term (6 months)</div>
                    <div className="text-2xl font-bold text-gray-900">75%</div>
                    <div className="text-sm text-gray-600 mt-1">National average target</div>
                  </div>
                  
                  <div className="p-4 bg-white rounded-lg border border-gray-200">
                    <div className="text-sm font-medium text-gray-600 mb-2">Medium-term (1 year)</div>
                    <div className="text-2xl font-bold text-gray-900">78%</div>
                    <div className="text-sm text-gray-600 mt-1">Target with interventions</div>
                  </div>
                  
                  <div className="p-4 bg-white rounded-lg border border-gray-200">
                    <div className="text-sm font-medium text-gray-600 mb-2">Long-term (2 years)</div>
                    <div className="text-2xl font-bold text-gray-900">82%</div>
                    <div className="text-sm text-gray-600 mt-1">Sustained excellence target</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EducationDashboard;