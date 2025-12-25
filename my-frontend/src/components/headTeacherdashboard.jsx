import React, { useState, useMemo } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, Award, Users, BookOpen, Target, ChevronDown, ChevronUp } from 'lucide-react';

const TeacherDashboard = () => {
  const [selectedSubject, setSelectedSubject] = useState('all');
  const [expandedSection, setExpandedSection] = useState(null);

  // Mock student data with multiple subjects and assignments
  const students = useMemo(() => {
    const names = ['Emma Wilson', 'Liam Chen', 'Sophia Martinez', 'Noah Anderson', 'Olivia Brown', 'Ethan Davis', 'Ava Garcia', 'Mason Rodriguez', 'Isabella Taylor', 'Lucas Moore', 'Mia Johnson', 'Elijah Lee', 'Charlotte White', 'James Harris', 'Amelia Clark', 'Benjamin Lewis', 'Harper Walker', 'Alexander Hall', 'Evelyn Allen', 'Daniel Young', 'Abigail King', 'Michael Wright', 'Emily Lopez', 'Matthew Hill', 'Elizabeth Scott', 'David Green', 'Sofia Adams', 'Joseph Baker', 'Avery Nelson', 'Samuel Carter'];
    
    return names.map((name, i) => {
      const mathGrades = Array.from({length: 5}, () => Math.floor(Math.random() * 30) + 70);
      const scienceGrades = Array.from({length: 5}, () => Math.floor(Math.random() * 30) + 65);
      const englishGrades = Array.from({length: 5}, () => Math.floor(Math.random() * 35) + 60);
      
      return {
        id: i + 1,
        name,
        math: {
          grades: mathGrades,
          average: mathGrades.reduce((a, b) => a + b, 0) / mathGrades.length,
        },
        science: {
          grades: scienceGrades,
          average: scienceGrades.reduce((a, b) => a + b, 0) / scienceGrades.length,
        },
        english: {
          grades: englishGrades,
          average: englishGrades.reduce((a, b) => a + b, 0) / englishGrades.length,
        }
      };
    });
  }, []);

  const subjects = ['math', 'science', 'english'];

  // Calculate analytics
  const analytics = useMemo(() => {
    const getSubjectData = (subject) => {
      const grades = students.map(s => s[subject].average);
      const avg = grades.reduce((a, b) => a + b, 0) / grades.length;
      const sorted = [...grades].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      
      return {
        average: avg,
        median,
        highest: Math.max(...grades),
        lowest: Math.min(...grades),
        passing: grades.filter(g => g >= 60).length,
        struggling: grades.filter(g => g < 70).length,
        excelling: grades.filter(g => g >= 90).length,
      };
    };

    if (selectedSubject === 'all') {
      const allGrades = students.flatMap(s => [s.math.average, s.science.average, s.english.average]);
      const avg = allGrades.reduce((a, b) => a + b, 0) / allGrades.length;
      
      return {
        overall: {
          average: avg,
          struggling: students.filter(s => 
            (s.math.average + s.science.average + s.english.average) / 3 < 70
          ).length,
          excelling: students.filter(s => 
            (s.math.average + s.science.average + s.english.average) / 3 >= 90
          ).length,
        },
        math: getSubjectData('math'),
        science: getSubjectData('science'),
        english: getSubjectData('english'),
      };
    } else {
      return {
        [selectedSubject]: getSubjectData(selectedSubject)
      };
    }
  }, [students, selectedSubject]);

  // Get struggling students
  const strugglingStudents = useMemo(() => {
    return students
      .map(s => ({
        ...s,
        overallAvg: (s.math.average + s.science.average + s.english.average) / 3,
        lowestSubject: [
          { name: 'Math', avg: s.math.average },
          { name: 'Science', avg: s.science.average },
          { name: 'English', avg: s.english.average }
        ].sort((a, b) => a.avg - b.avg)[0]
      }))
      .filter(s => s.overallAvg < 75)
      .sort((a, b) => a.overallAvg - b.overallAvg)
      .slice(0, 5);
  }, [students]);

  // Get top performers
  const topPerformers = useMemo(() => {
    return students
      .map(s => ({
        ...s,
        overallAvg: (s.math.average + s.science.average + s.english.average) / 3
      }))
      .sort((a, b) => b.overallAvg - a.overallAvg)
      .slice(0, 5);
  }, [students]);

  // Grade distribution
  const getGradeDistribution = () => {
    const grades = selectedSubject === 'all' 
      ? students.flatMap(s => [s.math.average, s.science.average, s.english.average])
      : students.map(s => s[selectedSubject].average);

    return [
      { range: '90-100', count: grades.filter(g => g >= 90).length, color: 'bg-green-500' },
      { range: '80-89', count: grades.filter(g => g >= 80 && g < 90).length, color: 'bg-blue-500' },
      { range: '70-79', count: grades.filter(g => g >= 70 && g < 80).length, color: 'bg-yellow-500' },
      { range: '60-69', count: grades.filter(g => g >= 60 && g < 70).length, color: 'bg-orange-500' },
      { range: '<60', count: grades.filter(g => g < 60).length, color: 'bg-red-500' },
    ];
  };

  const StatCard = ({ title, value, subtitle, icon: Icon, trend, color = 'blue' }) => (
    <div className="bg-white rounded-lg shadow p-6 border-l-4" style={{ borderLeftColor: `var(--tw-${color}-500)` }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 font-medium">{title}</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-full`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
      {trend && (
        <div className="mt-4 flex items-center text-sm">
          {trend > 0 ? (
            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
          )}
          <span className={trend > 0 ? 'text-green-600' : 'text-red-600'}>
            {Math.abs(trend)}% from last month
          </span>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Class Dashboard</h1>
          <p className="text-gray-600 mt-2">Overview of your class performance and insights</p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Class Average"
            value={selectedSubject === 'all' 
              ? analytics.overall.average.toFixed(1)
              : analytics[selectedSubject].average.toFixed(1)}
            subtitle="Overall performance"
            icon={Target}
            trend={2.5}
            color="blue"
          />
          <StatCard
            title="Students Excelling"
            value={selectedSubject === 'all'
              ? analytics.overall.excelling
              : analytics[selectedSubject].excelling}
            subtitle="Grade ≥ 90%"
            icon={Award}
            color="green"
          />
          <StatCard
            title="Need Support"
            value={selectedSubject === 'all'
              ? analytics.overall.struggling
              : analytics[selectedSubject].struggling}
            subtitle="Grade < 70%"
            icon={AlertCircle}
            color="orange"
          />
          <StatCard
            title="Total Students"
            value="30"
            subtitle="Class size"
            icon={Users}
            color="purple"
          />
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Students Needing Attention */}
          <div className="bg-white rounded-lg shadow">
            <button
              onClick={() => setExpandedSection(expandedSection === 'struggling' ? null : 'struggling')}
              className="w-full p-6 flex items-center justify-between text-left"
            >
              <h2 className="text-xl font-bold text-gray-900 flex items-center">
                <AlertCircle className="w-5 h-5 mr-2 text-orange-500" />
                Priority Students
              </h2>
              {expandedSection === 'struggling' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            {(expandedSection === 'struggling' || expandedSection === null) && (
              <div className="px-6 pb-6">
                <p className="text-sm text-gray-600 mb-4">Students who need extra support</p>
                <div className="space-y-3">
                  {strugglingStudents.map((student) => (
                    <div key={student.id} className="border border-orange-200 bg-orange-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-900">{student.name}</span>
                        <span className="text-sm font-medium text-orange-700">
                          Avg: {student.overallAvg.toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-sm text-gray-600">
                        <span className="font-medium">Weakest:</span> {student.lowestSubject.name} ({student.lowestSubject.avg.toFixed(1)}%)
                      </div>
                      <div className="mt-2 flex gap-2">
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          Math: {student.math.average.toFixed(0)}%
                        </span>
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          Science: {student.science.average.toFixed(0)}%
                        </span>
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          English: {student.english.average.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Top Performers */}
          <div className="bg-white rounded-lg shadow">
            <button
              onClick={() => setExpandedSection(expandedSection === 'top' ? null : 'top')}
              className="w-full p-6 flex items-center justify-between text-left"
            >
              <h2 className="text-xl font-bold text-gray-900 flex items-center">
                <Award className="w-5 h-5 mr-2 text-green-500" />
                Top Performers
              </h2>
              {expandedSection === 'top' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            {(expandedSection === 'top' || expandedSection === null) && (
              <div className="px-6 pb-6">
                <p className="text-sm text-gray-600 mb-4">Students excelling in the class</p>
                <div className="space-y-3">
                  {topPerformers.map((student, index) => (
                    <div key={student.id} className="border border-green-200 bg-green-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold mr-3">
                            {index + 1}
                          </span>
                          <span className="font-semibold text-gray-900">{student.name}</span>
                        </div>
                        <span className="text-sm font-medium text-green-700">
                          Avg: {student.overallAvg.toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-2 flex gap-2">
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          Math: {student.math.average.toFixed(0)}%
                        </span>
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          Science: {student.science.average.toFixed(0)}%
                        </span>
                        <span className="px-2 py-1 bg-white rounded text-xs">
                          English: {student.english.average.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Subject Comparison (only show when "all" is selected) */}
        {selectedSubject === 'all' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Subject Comparison</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {subjects.map(subject => (
                <div key={subject} className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold capitalize mb-4 text-gray-900">{subject}</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Average:</span>
                      <span className="font-semibold">{analytics[subject].average.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Median:</span>
                      <span className="font-semibold">{analytics[subject].median.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Highest:</span>
                      <span className="font-semibold text-green-600">{analytics[subject].highest.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Lowest:</span>
                      <span className="font-semibold text-red-600">{analytics[subject].lowest.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between pt-2 border-t">
                      <span className="text-gray-600">Need Support:</span>
                      <span className="font-semibold text-orange-600">{analytics[subject].struggling}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TeacherDashboard;