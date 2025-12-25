import React, { useState } from 'react';
import { Book, Users, GraduationCap, ChevronDown, ChevronRight, Award, TrendingUp } from 'lucide-react';

const SubjectTeacherDashboard = () => {
  // Mock data - In production, this would come from API based on logged-in teacher
  const teacherData = {
    teacherId: 1,
    teacherName: "Sarah Johnson",
    teacherEmail: "sarah.johnson@school.edu",
    photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah"
  };

  const assignedClasses = [
    {
      classId: 1,
      className: "Class 10-A",
      gradeId: 10,
      gradeName: "Grade 10",
      schoolId: 1,
      subjects: [
        {
          subjectId: 1,
          subjectName: "Mathematics",
          subjectCode: "MATH10",
          students: [
            { studentId: 1, studentName: "Alice Cooper", rollId: "10A-001", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Alice", score: 85 },
            { studentId: 2, studentName: "Bob Smith", rollId: "10A-002", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Bob", score: 92 },
            { studentId: 3, studentName: "Charlie Brown", rollId: "10A-003", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Charlie", score: 78 }
          ]
        },
        {
          subjectId: 2,
          subjectName: "Physics",
          subjectCode: "PHY10",
          students: [
            { studentId: 1, studentName: "Alice Cooper", rollId: "10A-001", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Alice", score: 88 },
            { studentId: 2, studentName: "Bob Smith", rollId: "10A-002", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Bob", score: 95 },
            { studentId: 3, studentName: "Charlie Brown", rollId: "10A-003", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Charlie", score: 82 }
          ]
        }
      ]
    },
    {
      classId: 2,
      className: "Class 10-B",
      gradeId: 10,
      gradeName: "Grade 10",
      schoolId: 1,
      subjects: [
        {
          subjectId: 1,
          subjectName: "Mathematics",
          subjectCode: "MATH10",
          students: [
            { studentId: 4, studentName: "Diana Prince", rollId: "10B-001", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Diana", score: 91 },
            { studentId: 5, studentName: "Ethan Hunt", rollId: "10B-002", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Ethan", score: 87 },
            { studentId: 6, studentName: "Fiona Green", rollId: "10B-003", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Fiona", score: 94 }
          ]
        }
      ]
    },
    {
      classId: 3,
      className: "Class 9-A",
      gradeId: 9,
      gradeName: "Grade 9",
      schoolId: 1,
      subjects: [
        {
          subjectId: 3,
          subjectName: "Mathematics",
          subjectCode: "MATH9",
          students: [
            { studentId: 7, studentName: "George Wilson", rollId: "9A-001", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=George", score: 76 },
            { studentId: 8, studentName: "Hannah Lee", rollId: "9A-002", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Hannah", score: 89 },
            { studentId: 9, studentName: "Ian Malcolm", rollId: "9A-003", photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Ian", score: 83 }
          ]
        }
      ]
    }
  ];

  const [expandedClass, setExpandedClass] = useState(null);
  const [expandedSubject, setExpandedSubject] = useState(null);
  const [selectedView, setSelectedView] = useState('overview');

  const toggleClass = (classId) => {
    setExpandedClass(expandedClass === classId ? null : classId);
    setExpandedSubject(null);
  };

  const toggleSubject = (subjectId) => {
    setExpandedSubject(expandedSubject === subjectId ? null : subjectId);
  };

  const getGradeColor = (score) => {
    if (score >= 90) return 'text-green-600 bg-green-50';
    if (score >= 75) return 'text-blue-600 bg-blue-50';
    if (score >= 60) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const calculateClassAverage = (subjects) => {
    let totalScore = 0;
    let totalStudents = 0;
    subjects.forEach(subject => {
      subject.students.forEach(student => {
        totalScore += student.score || 0;
        totalStudents++;
      });
    });
    return totalStudents > 0 ? (totalScore / totalStudents).toFixed(1) : 0;
  };

  const calculateSubjectAverage = (students) => {
    const total = students.reduce((sum, student) => sum + (student.score || 0), 0);
    return students.length > 0 ? (total / students.length).toFixed(1) : 0;
  };

  const totalStudents = assignedClasses.reduce((sum, cls) => 
    sum + cls.subjects.reduce((subSum, subject) => subSum + subject.students.length, 0), 0
  );

  const totalSubjects = assignedClasses.reduce((sum, cls) => sum + cls.subjects.length, 0);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-4">
              <img 
                src={teacherData.photo} 
                alt={teacherData.teacherName}
                className="w-12 h-12 rounded-full border-2 border-indigo-100"
              />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{teacherData.teacherName}</h1>
                <p className="text-sm text-gray-500">{teacherData.teacherEmail}</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedView('overview')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedView === 'overview' 
                    ? 'bg-indigo-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setSelectedView('classes')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedView === 'classes' 
                    ? 'bg-indigo-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                My Classes
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {selectedView === 'overview' && (
          <>
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Classes</p>
                    <p className="text-3xl font-bold text-gray-900 mt-2">{assignedClasses.length}</p>
                  </div>
                  <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
                    <GraduationCap className="w-6 h-6 text-indigo-600" />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Subjects</p>
                    <p className="text-3xl font-bold text-gray-900 mt-2">{totalSubjects}</p>
                  </div>
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                    <Book className="w-6 h-6 text-blue-600" />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Students</p>
                    <p className="text-3xl font-bold text-gray-900 mt-2">{totalStudents}</p>
                  </div>
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                    <Users className="w-6 h-6 text-green-600" />
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Stats by Class */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-indigo-600" />
                Class Performance Overview
              </h2>
              <div className="space-y-4">
                {assignedClasses.map(cls => (
                  <div key={cls.classId} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                        <GraduationCap className="w-5 h-5 text-indigo-600" />
                      </div>
                      <div>
                        <p className="font-semibold text-gray-900">{cls.className}</p>
                        <p className="text-sm text-gray-500">{cls.subjects.length} subject(s)</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm text-gray-600">Average Score</p>
                        <p className="text-2xl font-bold text-indigo-600">{calculateClassAverage(cls.subjects)}%</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {selectedView === 'classes' && (
          <div className="space-y-6">
            {assignedClasses.map(cls => (
              <div key={cls.classId} className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                {/* Class Header */}
                <div 
                  onClick={() => toggleClass(cls.classId)}
                  className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
                      <GraduationCap className="w-6 h-6 text-indigo-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{cls.className}</h3>
                      <p className="text-sm text-gray-500">{cls.gradeName} • {cls.subjects.length} Subject(s)</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Class Average</p>
                      <p className="text-xl font-bold text-indigo-600">{calculateClassAverage(cls.subjects)}%</p>
                    </div>
                    {expandedClass === cls.classId ? (
                      <ChevronDown className="w-6 h-6 text-gray-400" />
                    ) : (
                      <ChevronRight className="w-6 h-6 text-gray-400" />
                    )}
                  </div>
                </div>

                {/* Subjects */}
                {expandedClass === cls.classId && (
                  <div className="border-t border-gray-200 bg-gray-50">
                    {cls.subjects.map(subject => (
                      <div key={subject.subjectId} className="border-b border-gray-200 last:border-b-0">
                        <div 
                          onClick={() => toggleSubject(subject.subjectId)}
                          className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-100 transition-colors"
                        >
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                              <Book className="w-5 h-5 text-blue-600" />
                            </div>
                            <div>
                              <p className="font-medium text-gray-900">{subject.subjectName}</p>
                              <p className="text-sm text-gray-500">{subject.subjectCode} • {subject.students.length} Students</p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-4">
                            <div className="text-right">
                              <p className="text-sm text-gray-600">Subject Average</p>
                              <p className="text-lg font-bold text-blue-600">{calculateSubjectAverage(subject.students)}%</p>
                            </div>
                            {expandedSubject === subject.subjectId ? (
                              <ChevronDown className="w-5 h-5 text-gray-400" />
                            ) : (
                              <ChevronRight className="w-5 h-5 text-gray-400" />
                            )}
                          </div>
                        </div>

                        {/* Students */}
                        {expandedSubject === subject.subjectId && (
                          <div className="bg-white p-4">
                            <div className="space-y-2">
                              {subject.students.map(student => (
                                <div key={student.studentId} className="flex items-center justify-between p-3 rounded-lg border border-gray-200 hover:border-indigo-300 transition-colors">
                                  <div className="flex items-center space-x-3">
                                    <img 
                                      src={student.photo} 
                                      alt={student.studentName}
                                      className="w-10 h-10 rounded-full border-2 border-gray-200"
                                    />
                                    <div>
                                      <p className="font-medium text-gray-900">{student.studentName}</p>
                                      <p className="text-sm text-gray-500">Roll No: {student.rollId}</p>
                                    </div>
                                  </div>
                                  <div className="flex items-center space-x-3">
                                    <div className="text-right">
                                      <p className="text-sm text-gray-600">Score</p>
                                      <p className={`text-lg font-bold px-3 py-1 rounded-lg ${getGradeColor(student.score)}`}>
                                        {student.score}%
                                      </p>
                                    </div>
                                    <Award className={`w-5 h-5 ${student.score >= 90 ? 'text-yellow-500' : 'text-gray-300'}`} />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SubjectTeacherDashboard;