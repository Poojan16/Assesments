import React, { useState, useMemo, useEffect } from 'react';
import { TrendingUp, TrendingDown, Award, Users, BarChart3, Grid3x3, ChevronDown } from 'lucide-react';
import { initializeAuth } from '../authSlice';
import { useDispatch, useSelector } from 'react-redux';

const ClassSummaryReport = () => {
  const [selectedClass, setSelectedClass] = useState('');
  const [classes, setClasses] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [allStudents, setAllStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('overview');
  const [sortBy, setSortBy] = useState('overall');
  const [currentPage, setCurrentPage] = useState(1);
  const studentsPerPage = 8;
  const [scores, setScores] = useState([]);
  const {user} = useSelector((state) => state.auth);
  const dispatch = useDispatch();
  useEffect(() => {
      // Initialize auth from localStorage on app load
      dispatch(initializeAuth());
    }, [dispatch]);

  const backend_url = process.env.REACT_APP_BACKEND_URL;

  const [teacher, setTeacher] = useState(null);

  useEffect(() => {
    const fetchTeacher = async (user) => {
      try {
        if (!user?.userEmail) return;
        
        const res = await fetch(
          `${backend_url}/teachers/email?email=${user.userEmail}`
        );
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        setTeacher(data?.data);
      } catch (err) {
        console.error("Error fetching teacher:", err);
      }
    }
    
    fetchTeacher(user);
  }, [user]);

  // Fetch initial data
  useEffect(() => {
    const fetchData = async (teacher) => {
      try {
        setLoading(true);
        
        const [classesResponse, subjectsResponse, studentsResponse, scoreResponse] = await Promise.all([
          fetch(`${backend_url}/classes/`),
          fetch(`${backend_url}/subjects/`),
          fetch(`${backend_url}/students/`),
          fetch(`${backend_url}/students/score`)
        ]);
        
        if (!classesResponse.ok) throw new Error(`HTTP error! status: ${classesResponse.status}`);
        if (!subjectsResponse.ok) throw new Error(`HTTP error! status: ${subjectsResponse.status}`);
        if (!studentsResponse.ok) throw new Error(`HTTP error! status: ${studentsResponse.status}`);
        if (!scoreResponse.ok) throw new Error(`HTTP error! status: ${scoreResponse.status}`);
        
        const sampleClasses = await classesResponse.json();
        const sampleSubjects = await subjectsResponse.json();
        const sampleStudents = await studentsResponse.json();
        const sampleScores = await scoreResponse.json();

        console.log('Fetched data:', {
          classes: sampleClasses?.data,
          subjects: sampleSubjects?.data,
          students: sampleStudents?.data,
          scores: sampleScores?.data
        });

        // Set all data
        setClasses((sampleClasses?.data).filter(c => c.schoolId === teacher?.schoolId) || []);
        setSubjects(sampleSubjects?.data || []);
        setAllStudents(sampleStudents?.data || []);
        setScores(sampleScores?.data || []);
        
        // Only set selectedClass after all data is loaded
        if (sampleClasses?.data?.length > 0) {
          setSelectedClass(sampleClasses.data[0].classId);
        }
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData(teacher);
  }, [teacher]);

  // Calculate students in selected class
  const classStudents = useMemo(() => {
    if (!selectedClass || !allStudents.length) return [];
    
    const filtered = allStudents.filter(student => student.schoolId === teacher?.schoolId && student.classId === Number(selectedClass));
    console.log(`Class ${selectedClass} students:`, filtered);
    return filtered;
  }, [selectedClass, allStudents]);

  console.log('Class students:',classStudents);

  // Calculate class average
  const classAvg = useMemo(() => {
    if (!selectedClass || !scores.length || !classStudents.length) return 0;
  
    // Get student IDs in selected class
    const classStudentIds = classStudents.map(student => student.studentId);
    
    // Get scores for those students
    const classScores = scores.filter(score =>
      classStudentIds.includes(score.studentId)
    );
  
    console.log(`Class ${selectedClass} scores:`, classScores);
    
    if (classScores.length === 0) return 0;
  
    const total = classScores.reduce(
      (sum, s) => sum + Number(s.score || 0),
      0
    );
  
    const avg = (total / classScores.length).toFixed(2);
    console.log(`Class ${selectedClass} average:`, avg);
    return avg;
  }, [selectedClass, scores, classStudents]);

  console.log('Class average:',classAvg);

  // Calculate statistics for selected class
  const stats = useMemo(() => {
    console.log('Calculating stats with:', {
      selectedClass,
      scoresCount: scores.length,
      classStudentsCount: classStudents.length,
      subjectsCount: subjects.length
    });

    if (!selectedClass || !scores.length || !classStudents.length || !subjects.length) {
      return { 
        classAvg: 0, 
        subjectAvgs: [], 
        top10: [], 
        needsAttention: [],
        sorted: [] 
      };
    }

    const studentIds = classStudents.map(s => s.studentId);

    // Calculate subject averages
    const subjectAvgs = subjects.map(subject => {
      const subjectScores = scores.filter(
        score =>
          score.subjectId === subject.subjectId &&
          studentIds.includes(score.studentId)
      );

      const total = subjectScores.reduce(
        (sum, score) => sum + Number(score.score || 0),
        0
      );

      return subjectScores.length ? total / subjectScores.length : 0;
    });

    console.log('Subject averages:', subjectAvgs);

    // Calculate overall scores for each student
    const studentsWithScores = classStudents.map(student => {
      const studentScores = scores.filter(s => s.studentId === student.studentId);
      const overall = studentScores.length > 0 
        ? studentScores.reduce((sum, s) => sum + Number(s.score || 0), 0) / studentScores.length
        : 0;
      
      // Map scores by subject
      const studentSubjectScores = subjects.map(subject => {
        const subjectScore = studentScores.find(s => s.subjectId === subject.subjectId);
        return subjectScore ? Number(subjectScore.score) : 0;
      });
      
      return {
        ...student,
        overall: Number(overall.toFixed(2)),
        scores: studentSubjectScores
      };
    });

    console.log('Students with scores:', studentsWithScores);

    // Sort students by overall score
    const sorted = [...studentsWithScores].sort((a, b) => b.overall - a.overall);
    const top10 = sorted.slice(0, Math.min(10, sorted.length));
    const needsAttention = sorted.slice(-Math.min(5, sorted.length));

    return { 
      classAvg: Number(classAvg), 
      subjectAvgs, 
      top10, 
      needsAttention,
      sorted 
    };
  }, [selectedClass, scores, classStudents, subjects, classAvg]);

  // Sort students for display
  const sortedStudents = useMemo(() => {
    if (!stats.sorted || !stats.sorted.length) return [];
    
    const sorted = [...stats.sorted];
    if (sortBy === 'overall') {
      return sorted.sort((a, b) => b.overall - a.overall);
    } else if (sortBy === 'attendance') {
      return sorted.sort((a, b) => (b.attendance || 0) - (a.attendance || 0));
    } else {
      // Sort by specific subject
      const subject = subjects.find(s => s.subjectId === sortBy);
      if (!subject) return sorted;
      
      const subjectIndex = subjects.indexOf(subject);
      return sorted.sort((a, b) => 
        (b.scores?.[subjectIndex] || 0) - (a.scores?.[subjectIndex] || 0)
      );
    }
  }, [stats.sorted, sortBy, subjects]);

  // Pagination
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstStudent = indexOfLastStudent - studentsPerPage;
  const currentStudents = sortedStudents.slice(indexOfFirstStudent, indexOfLastStudent);
  const totalPages = Math.ceil(sortedStudents.length / studentsPerPage);

  const getGradeColor = (score) => {
    if (score >= 90) return 'bg-green-500';
    if (score >= 75) return 'bg-blue-500';
    if (score >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getGrade = (score) => {
    if (score >= 90) return 'A';
    if (score >= 75) return 'B';
    if (score >= 60) return 'C';
    return 'D';
  };

  const selectedClassInfo = classes.find(c => c.classId === selectedClass);

  // Debug logs
  console.log('Current state:', {
    selectedClass,
    selectedClassInfo,
    classesCount: classes.length,
    studentsCount: allStudents.length,
    classStudentsCount: classStudents.length,
    scoresCount: scores.length,
    subjectsCount: subjects.length,
    stats,
    sortedStudentsCount: sortedStudents.length,
    currentStudentsCount: currentStudents.length
  });

  // Reset current page when class changes
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedClass]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-2xl font-semibold text-indigo-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header with Class Selector */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">Class Summary Report</h1>
              <p className="text-gray-600">Class: {selectedClassInfo?.className || 'No class selected'}</p>
            </div>
            
            {/* Class Selector */}
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Class
              </label>
              <div className="relative">
                <select
                  value={selectedClass}
                  onChange={(e) => {
                    console.log('Changing class to:', e.target.value);
                    setSelectedClass(e.target.value);
                  }}
                  className="appearance-none w-full md:w-64 px-4 py-3 pr-10 bg-indigo-50 border-2 border-indigo-200 rounded-lg font-semibold text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent cursor-pointer"
                  disabled={classes.length === 0}
                >
                  {classes.length === 0 ? (
                    <option value="">No classes available</option>
                  ) : (
                    classes.map(cls => (
                      <option key={cls.classId} value={cls.classId}>
                        {cls.className}
                      </option>
                    ))
                  )}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-indigo-600 pointer-events-none" size={20} />
              </div>
            </div>
          </div>
          
          {/* Class Info */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex flex-wrap gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Users className="text-indigo-600" size={18} />
                <span className="font-medium">Total Students: {classStudents.length}</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="text-indigo-600" size={18} />
                <span className="font-medium">Class Average: {stats.classAvg}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* View Toggle */}
        <div className="flex gap-2 mb-6 overflow-x-auto">
          <button
            onClick={() => setViewMode('overview')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
              viewMode === 'overview'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            <BarChart3 size={18} />
            Overview
          </button>
          <button
            onClick={() => setViewMode('grid')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
              viewMode === 'grid'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Grid3x3 size={18} />
            Grid View
          </button>
        </div>

        {/* Overview Mode */}
        {viewMode === 'overview' && (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Class Average</p>
                    <p className="text-3xl font-bold text-indigo-600 mt-1">{stats.classAvg}%</p>
                  </div>
                  <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center">
                    <BarChart3 className="text-indigo-600" size={24} />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Top Performer</p>
                    <p className="text-xl font-bold text-amber-600 mt-1">
                      {stats.top10[0]?.studentName || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-500">{stats.top10[0]?.overall || 0}%</p>
                  </div>
                  <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center">
                    <Award className="text-amber-600" size={24} />
                  </div>
                </div>
              </div>
            </div>

            {/* Subject Performance */}
            {subjects.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Subject-wise Performance</h2>
                <div className="space-y-4">
                  {subjects.map((subject, idx) => (
                    <div key={subject.subjectId}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-gray-700">{subject.subjectName}</span>
                        <span className="font-bold text-indigo-600">
                          {stats.subjectAvgs[idx] ? stats.subjectAvgs[idx].toFixed(2) : 0}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full transition-all ${getGradeColor(stats.subjectAvgs[idx] || 0)}`}
                          style={{ width: `${Math.min(stats.subjectAvgs[idx] || 0, 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top 10 Students */}
            {stats.top10.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <Award className="text-amber-500" />
                  Top Performers
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-2 text-gray-600 font-semibold">Name</th>
                        {subjects.map(subject => (
                          <th key={subject.subjectId} className="text-center py-3 px-2 text-gray-600 font-semibold">
                            {subject.subjectName}
                          </th>
                        ))}
                        <th className="text-center py-3 px-2 text-gray-600 font-semibold">Overall</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.top10.map((student, index) => (
                        <tr key={student.studentId || index} className="border-b hover:bg-gray-50">
                          <td className="py-3 px-2 font-medium">{student.studentName}</td>
                          {student.scores?.map((score, idx) => (
                            <td key={idx} className="py-3 px-2 text-center">
                              <span className={`px-2 py-1 rounded font-semibold text-white ${getGradeColor(score)}`}>
                                {score}
                              </span>
                            </td>
                          ))}
                          <td className="py-3 px-2 text-center">
                            <span className="font-bold text-indigo-600 text-lg">{student.overall}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Students Needing Attention */}
            {stats.needsAttention.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-bold text-red-600 mb-4">Students Need Attention</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {stats.needsAttention.map((student, index) => (
                    <div key={student.studentId || index} className="border border-red-200 rounded-lg p-4 bg-red-50">
                      <p className="font-semibold text-gray-800">{student.studentName}</p>
                      <p className="text-2xl font-bold text-red-600 mt-1">{student.overall}%</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Grid View */}
        {viewMode === 'grid' && (
          <>
            {currentStudents.length > 0 ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {currentStudents.map((student, index) => (
                    <div key={student.studentId || index} className="bg-white rounded-lg shadow-lg p-5 hover:shadow-xl transition">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="font-bold text-gray-800">{student.studentName}</h3>
                          <p className="text-sm text-gray-500">ID: {student.studentId}</p>
                        </div>
                      </div>
                      
                      <div className="mb-3">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm text-gray-600">Overall Score</span>
                          <span className={`px-2 py-1 rounded font-bold text-white text-sm ${getGradeColor(student.overall)}`}>
                            {getGrade(student.overall)}
                          </span>
                        </div>
                        <p className="text-3xl font-bold text-indigo-600">{student.overall}%</p>
                      </div>

                      {subjects.length > 0 && (
                        <div className="space-y-2 mb-3">
                          {subjects.map((subject, idx) => (
                            <div key={subject.subjectId} className="flex justify-between items-center text-sm">
                              <span className="text-gray-600">{subject.subjectName}</span>
                              <span className="font-semibold">{student.scores?.[idx] || 0}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex justify-center items-center gap-4 mt-6">
                    <button
                      className="px-4 py-2 bg-indigo-500 text-white rounded-md hover:bg-indigo-600 transition duration-200 disabled:bg-gray-300 disabled:cursor-not-allowed"
                      onClick={() => setCurrentPage(currentPage - 1)}
                      disabled={currentPage === 1}
                    >
                      Previous
                    </button>
                    <span className="text-gray-600 font-medium">
                      Page {currentPage} of {totalPages}
                    </span>
                    <button
                      className="px-4 py-2 bg-indigo-500 text-white rounded-md hover:bg-indigo-600 transition duration-200 disabled:bg-gray-300 disabled:cursor-not-allowed"
                      onClick={() => setCurrentPage(currentPage + 1)}
                      disabled={currentPage === totalPages}
                    >
                      Next
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white rounded-lg shadow-lg p-8 text-center">
                <p className="text-gray-600 text-lg">No students found in this class</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ClassSummaryReport;