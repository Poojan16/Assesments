import React, { useEffect, useRef, useState, useCallback } from 'react';
import { debounce } from 'lodash';
import { 
  Search, LogOut, Upload, Download, Plus, Edit, Trash2, X, Link, 
  FileText, Send, ZoomIn, ZoomOut, CheckCircle, XCircle, MessageSquare, 
  Filter, ChevronDown, ChevronUp, User, Phone, Mail, Calendar, 
  GraduationCap, Award, TrendingUp, TrendingDown, Check, X as XIcon, 
  MoreVertical, ChevronLeft, ChevronRight 
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Excel from 'exceljs';
import { saveAs } from 'file-saver';
import Loader from './loader';
import { useDispatch, useSelector } from 'react-redux';
import * as XLSX from 'xlsx';
import { initializeAuth, logout } from '../authSlice';

const HeadTeacher = () => {
  const { user } = useSelector((state) => state.auth);
  const dispatch = useDispatch();

  useEffect(() => {
    // Initialize auth from localStorage on app load
    dispatch(initializeAuth());
  }, [dispatch]);


  const [showProvisionModal, setShowProvisionModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isStudent, setIsStudent] = useState(true);
  const [successMessage, setSuccessMessage] = useState(''); 
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reports] = useState([
    { id: 1, subject: 'Mathematics', teacher: 'Mr. Smith', status: 'pending', comments: null, sentDate: '2024-12-01' },
    { id: 2, subject: 'Science', teacher: 'Ms. Johnson', status: 'approved', comments: null, sentDate: '2024-11-28' },
    { id: 3, subject: 'English', teacher: 'Mrs. Brown', status: 'pending', comments: 'Please revise the grading section', sentDate: '2024-12-02' },
  ]);

  const [classes, setClasses] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [teacherMap, setTeacherMap] = useState([]);
  const [studentScore, setStudentScore] = useState([]);
  const [schools, setSchools] = useState({});
  const [grades, setGrades] = useState([]);
  const [gradesWithPerformance, setGradesWithPerformance] = useState([]);
  
  // Filter states
  const [filters, setFilters] = useState({
    classId: '',
    gender: '',
    grade: '',
    performance: '',
    status: 'active'
  });
  
  const [showFilters, setShowFilters] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'name', direction: 'asc' });
  const [teacher, setTeacher] = useState({});
  const [teachers,setTeachers] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (!user?.userEmail) return;

        // 1. First fetch teacher by email
        const teacherRes = await fetch(
          `http://127.0.0.1:8000/teachers/email?email=${user.userEmail}`
        );
        const teacherData = await teacherRes.json();
        setTeacher(teacherData?.data);

        // 2. Only proceed if teacher data is valid
        if (!teacherData || !(teacherData?.data).teacherId) {
          setLoading(false);
          return;
        }

        setLoading(true);
        
        const [
          classesResponse, 
          subjectsResponse, 
          studentsResponse, 
          teacherResponse,
          teacherMapResponse, 
          studentScoreResponse, 
          schoolResponse, 
          gradeResponse
        ] = await Promise.all([
          fetch('http://127.0.0.1:8000/classes/'),
          fetch('http://127.0.0.1:8000/subjects/'),
          fetch('http://127.0.0.1:8000/students/'),
          fetch('http://127.0.0.1:8000/teachers/'),
          fetch(`http://127.0.0.1:8000/teachers/class_and_subjects?teacherId=${(teacherData?.data).teacherId}`),
          fetch('http://127.0.0.1:8000/students/score'),
          fetch(`http://127.0.0.1:8000/admin/schools/${(teacherData?.data)?.schoolId}`),
          fetch('http://127.0.0.1:8000/grades/')
        ]);

        const classesData = await classesResponse.json();
        const subjectsData = await subjectsResponse.json();
        const studentData = await studentsResponse.json();
        const teachersData = await teacherResponse.json();
        const teacherMapData = await teacherMapResponse.json();
        const studentScoreData = await studentScoreResponse.json();
        const schoolData = await schoolResponse.json();
        const gradeData = await gradeResponse.json();
    


        setSchools(schoolData?.data || {});
        setGrades(gradeData?.data || []);
        setSubjects(subjectsData?.data || []);
        setStudentScore(studentScoreData?.data || []);

        const filteredTeachers = (teachersData?.data || []).filter((teacherItem) => 
          teacherItem.schoolId === (teacherData?.data).schoolId 
        );
        setTeachers(filteredTeachers);

        if (classesData && subjectsData) {
          const filteredClasses = (classesData?.data || []).filter((classItem) => 
            classItem.schoolId === ((teacherData?.data).schoolId || 1)
          );
          setClasses(filteredClasses);

          const filteredStudent = (studentData?.data || []).filter((student) => 
            student.schoolId === ((teacherData?.data).schoolId || 1)
          );
          setStudents(filteredStudent);

          const filterTeacherMap = (teacherMapData?.data || []).filter((teacherItem) => 
            teacherItem.teacherId === teacherData.teacherId
          );
          setTeacherMap(filterTeacherMap);

          const gradesWithPerformance = (gradeData?.data || []).map(grade => ({
            ...grade,
            performance: getPerformance(grade.gradeLetter)
          }));
          setGradesWithPerformance(gradesWithPerformance);
        }
      } catch (err) {
        console.error("Error fetching data:", err);
        setErrorMessage("Failed to load data. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [user]);

  const getPerformance = (gradeLetter) => {
    switch (gradeLetter) {
      case 'A': return 'Excellent';
      case 'B': return 'Good';
      case 'C': return 'Average';
      case 'D': return 'Below Average';
      case 'E': return 'Poor';
      case 'N': return 'Not Graded';
      default: return 'Failed';
    }
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'approved': return 'bg-green-100 text-green-800';
      case 'rejected': return 'bg-red-100 text-red-800';
      default: return 'bg-yellow-100 text-yellow-800';
    }
  };

  const calculatePerformance = (totalScore) => {
    const grade = grades.find((grade) => 
      grade.upperLimit >= totalScore && grade.lowerLimit <= totalScore
    );
    
    return grade?.gradeLetter || "N/A";
  };

  const getPerformanceStatus = (totalScore) => {
    const gradeLetter = calculatePerformance(totalScore);
    return getPerformance(gradeLetter);
  };

  const getPerformanceColor = (performance) => {
    switch (performance) {
      case 'Excellent':
        return 'bg-green-500 text-white';
      case 'Good':
        return 'bg-yellow-500 text-black';
      case 'Average':
        return 'bg-orange-500 text-black';
      case 'Below Average':
        return 'bg-red-500 text-white';
      case 'Poor':
        return 'bg-purple-500 text-white';
      default:
        return 'bg-gray-200 text-gray-500';
    }
  };

  // avg score
  const getTotalScore = (studentId) => {
    const scores = studentScore.filter(score => score.studentId === studentId);
    const total =  scores.reduce((total, score) => total + score.score, 0);
    if(scores.length === 0) return 0
    return total/scores.length

  };

  const getSubjectScores = (studentId) => {
    return subjects.map(subject => {
      const score = studentScore.find(s => 
        s.studentId === studentId && s.subjectId === subject.subjectId
      );
      return {
        subject: subject.subjectName,
        score: score?.score || 0,
        maxScore: 100
      };
    });
  };

  // Filter and sort functions
  const filteredStudents = students.filter(student => {
    // Name search filter
    if (searchQuery && !student.studentName?.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    
    // Class filter
    if (filters.classId && student.classId !== parseInt(filters.classId)) {
      return false;
    }
    
    // Gender filter
    if (filters.gender && student.gender !== filters.gender) {
      return false;
    }
    
    // Grade filter
    if (filters.grade) {
      const totalScore = getTotalScore(student.studentId);
      const studentGrade = calculatePerformance(totalScore);
      if (studentGrade !== filters.grade) {
        return false;
      }
    }
    
    // Performance filter
    if (filters.performance) {
      const totalScore = getTotalScore(student.studentId);
      const performance = getPerformanceStatus(totalScore);
      if (performance !== filters.performance) {
        return false;
      }
    }
    
    // Status filter
    if (filters.status === 'inactive') {
      return false;
    }
    
    return true;
  });

  // Sorting function
  const sortedStudents = [...filteredStudents].sort((a, b) => {
    if (sortConfig.key === 'name') {
      return sortConfig.direction === 'asc' 
        ? (a.studentName || '').localeCompare(b.studentName || '')
        : (b.studentName || '').localeCompare(a.studentName || '');
    }
    
    if (sortConfig.key === 'score') {
      const scoreA = getTotalScore(a.studentId);
      const scoreB = getTotalScore(b.studentId);
      return sortConfig.direction === 'asc' ? scoreA - scoreB : scoreB - scoreA;
    }
    
    if (sortConfig.key === 'grade') {
      const gradeA = calculatePerformance(a);
      const gradeB = calculatePerformance(b);
      return sortConfig.direction === 'asc' 
        ? gradeA.localeCompare(gradeB)
        : gradeB.localeCompare(gradeA);
    }
    
    if (sortConfig.key === 'class') {
      const classA = classes.find(c => c.classId === a.classId)?.className || '';
      const classB = classes.find(c => c.classId === b.classId)?.className || '';
      return sortConfig.direction === 'asc' 
        ? classA.localeCompare(classB)
        : classB.localeCompare(classA);
    }
    
    return 0;
  });

  console.log('sorted',sortedStudents)

  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const clearFilters = () => {
    setFilters({
      classId: '',
      gender: '',
      grade: '',
      performance: '',
      status: 'active'
    });
    setSearchQuery('');
  };

  const getFilterCount = () => {
    let count = 0;
    if (filters.classId) count++;
    if (filters.gender) count++;
    if (filters.grade) count++;
    if (filters.performance) count++;
    if (filters.status !== 'active') count++;
    return count;
  };

  const handleExportExcel = () => {
    const exportToExcel = async (data, fileName) => {
      const workbook = new Excel.Workbook();
      const worksheet = workbook.addWorksheet('Students');
    
      // Define columns and headers
      worksheet.columns = [
        { header: 'Roll No', key: 'rollNo', width: 15 },
        { header: 'Student Name', key: 'name', width: 25 },
        { header: 'Gender', key: 'gender', width: 10 },
        { header: 'Date of Birth', key: 'dob', width: 15 },
        { header: 'Class', key: 'class', width: 15 },
        { header: 'Total Score', key: 'totalScore', width: 12 },
        { header: 'Grade', key: 'grade', width: 10 },
        { header: 'Performance', key: 'performance', width: 15 },
        { header: 'Address', key: 'address', width: 30 },
      ];
  
      const rowsToAdd = data.map(row => ({
        rollNo: row.rollId || '',
        name: row.studentName || '',
        gender: row.gender || '',
        dob: row.DOB ? new Date(row.DOB).toLocaleDateString() : '',
        class: classes.find(c => c.classId === row.classId)?.className || 'N/A',
        totalScore: getTotalScore(row.studentId),
        grade: calculatePerformance(getTotalScore(row.studentId)) === 'N' ? 'NG' : calculatePerformance(getTotalScore(row.studentId)),
        performance: getPerformanceStatus(getTotalScore(row.studentId)),
        address: `${row.address || ''}, ${row.city || ''}, ${row.state || ''}`.trim(),
      }));
    
      worksheet.addRows(rowsToAdd);
      
      // Define header row styles
      const headerRow = worksheet.getRow(1);
      headerRow.eachCell((cell) => {
        cell.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Arial', size: 12 };
        cell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: '1A47C2' },
        };
        cell.alignment = { horizontal: 'center', vertical: 'middle' };
        cell.border = {
          top: { style: 'medium' },
          left: { style: 'medium' },
          bottom: { style: 'medium' },
          right: { style: 'medium' }
        };
      });
      
      // freezing header
      worksheet.views = [
        {
          state: 'frozen',
          xSplit: 0,
          ySplit: 1, 
          activeCell: 'A2',
          showGridLines: true
        }
      ];
  
      // Style data rows
      worksheet.eachRow({ includeEmpty: false }, (row, rowNumber) => {
        if (rowNumber > 1) {
          row.eachCell((cell) => {
            cell.font = { name: 'Arial', size: 11 };
            cell.border = {
              top: { style: 'thin' },
              left: { style: 'thin' },
              bottom: { style: 'thin' },
              right: { style: 'thin' }
            };
            
            // Set alignment based on column
            const colLetter = cell.address.substring(0, 1);
            if (['A', 'C', 'D', 'E', 'F', 'G'].includes(colLetter)) {
              cell.alignment = { vertical: 'middle', horizontal: 'center' };
            } else {
              cell.alignment = { vertical: 'middle', horizontal: 'left' };
            }
          });
          
          // Style Total Score column
          const totalScoreCell = row.getCell('totalScore');
          const totalScore = totalScoreCell.value;
          if (totalScore !== undefined && totalScore !== null) {
            if (totalScore >= 90) {
              totalScoreCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'D1FAE5' } };
              totalScoreCell.font = { bold: true, color: { argb: '065F46' } };
            } else if (totalScore >= 80) {
              totalScoreCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'DCFCE7' } };
              totalScoreCell.font = { bold: true, color: { argb: '166534' } };
            } else if (totalScore >= 70) {
              totalScoreCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEF9C3' } };
              totalScoreCell.font = { bold: true, color: { argb: '854D0E' } };
            } else if (totalScore >= 60) {
              totalScoreCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFEDD5' } };
              totalScoreCell.font = { bold: true, color: { argb: '9A3412' } };
            } else if (totalScore < 60) {
              totalScoreCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEE2E2' } };
              totalScoreCell.font = { bold: true, color: { argb: '991B1B' } };
            }
          }
          
          // Style Grade column
          const gradeCell = row.getCell('grade');
          const grade = gradeCell.value;
          if (grade) {
            switch(grade) {
              case 'A':
              case 'A+':
                gradeCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'D1FAE5' } };
                gradeCell.font = { bold: true, color: { argb: '065F46' } };
                break;
              case 'B':
              case 'B+':
                gradeCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'DCFCE7' } };
                gradeCell.font = { bold: true, color: { argb: '166534' } };
                break;
              case 'C':
              case 'C+':
                gradeCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEF9C3' } };
                gradeCell.font = { bold: true, color: { argb: '854D0E' } };
                break;
              case 'D':
              case 'D+':
                gradeCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFEDD5' } };
                gradeCell.font = { bold: true, color: { argb: '9A3412' } };
                break;
              case 'F':
              case 'NG':
                gradeCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEE2E2' } };
                gradeCell.font = { bold: true, color: { argb: '991B1B' } };
                break;
            }
          }
          
          // Style Performance column
          const performanceCell = row.getCell('performance');
          const performance = performanceCell.value;
          
          switch(performance) {
            case 'Excellent':
              performanceCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'D1FAE5' } };
              performanceCell.font = { bold: true, color: { argb: '065F46' } };
              break;
            case 'Good':
              performanceCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'DCFCE7' } };
              performanceCell.font = { bold: true, color: { argb: '166534' } };
              break;
            case 'Average':
              performanceCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEF9C3' } };
              performanceCell.font = { bold: true, color: { argb: '854D0E' } };
              break;
            case 'Below Average':
              performanceCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFEDD5' } };
              performanceCell.font = { bold: true, color: { argb: '9A3412' } };
              break;
            case 'Poor':
              performanceCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FEE2E2' } };
              performanceCell.font = { bold: true, color: { argb: '991B1B' } };
              break;
          }
        }
      });
  
      // Auto fit columns
      worksheet.columns.forEach(column => {
        let maxLength = 0;
        column.eachCell({ includeEmpty: true }, cell => {
          const columnLength = cell.value ? cell.value.toString().length : 10;
          if (columnLength > maxLength) {
            maxLength = columnLength;
          }
        });
        column.width = maxLength < 10 ? 10 : maxLength + 2;
      });
  
      const buffer = await workbook.xlsx.writeBuffer();
      const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      saveAs(blob, `${fileName}.xlsx`);
    };
    exportToExcel(sortedStudents, "student_detailed_report");
  };

  const [showReportsList] = useState(false);
  const [selectedReport, setSelectedReport] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [comment, setComment] = useState('');

  const handleReportClick = (report) => {
    setSelectedReport(report);
    setZoomLevel(100);
    setComment(report.comments || '');
  };

  const handleApprove = () => {
    setSelectedReport(null);
    setComment('');
  };

  const handleReject = () => {
    if (!comment.trim()) {
      alert('Please add comments before rejecting');
      return;
    }
    setSelectedReport(null);
    setComment('');
  };

  const [studentForm, setStudentForm] = useState({
    studentId: '',
    changedBy: null,
    provision: ''
  });
    
  const [teacherForm, setTeacherForm] = useState({
    teacherId: '',
    changedBy:  null,
    offBoardingDate: null,
    provision: ''
  });
    
  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      dispatch(logout());
    }
  };
        
  // Template configuration for student import
  const [importValidationErrors, setImportValidationErrors] = useState([]);
  const [showImportModal, setShowImportModal] = useState(false);
  const STUDENT_TEMPLATE = {
    columns: [
      'Student Name',
      'DOB',
      'Gender',
      'Class',
      'Address',
      'City',
      'State',
      'Country',
      'Pincode',
      'Grade',
      'Parent Name',
      'Parent Email',
      'SchoolId'
    ],
    validations: {
      'Student Name': { type: 'string', maxLength: 100, required: true, pattern: /^[A-Za-z\s.'-]+$/ },
      'DOB': { 
        type: 'date', 
        required: true,
        minDate: new Date('1990-01-01'),
        maxDate: new Date()
      },
      'Gender': { 
        type: 'string', 
        required: true, 
        allowedValues: ['male', 'female'] 
      },
      'Class': { 
        type: 'string', 
        required: true,
        pattern: /^[A-Za-z0-9\s]+$/
      },
      'SchoolId': { type: 'number', required: true },
      'Address': { type: 'string', maxLength: 200, required: true },
      'City': { type: 'string', maxLength: 50, required: true, pattern: /^[A-Za-z\s]+$/ },
      'State': { type: 'string', maxLength: 50, required: true, pattern: /^[A-Za-z\s]+$/ },
      'Country': { type: 'string', maxLength: 50, required: true, pattern: /^[A-Za-z\s]+$/ },
      'Pincode': { type: 'string', required: true, pattern: /^\d{5,6}$/ },
      'Grade': { 
        type: 'string', 
        required: true,
        pattern: /^[A-FN]$/i
      },
      'Parent Name': { type: 'string', maxLength: 100, required: true },
      'Parent Email': { 
        type: 'string', 
        required: true, 
        pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/ 
      }
    }
  };
  
  const validateStudentExcelData = (excelData) => {
    const errors = [];
    
    if (!Array.isArray(excelData) || excelData.length === 0) {
      errors.push('No data found in the Excel file');
      return { isValid: false, errors, validatedData: null };
    }

    // Get headers from first row
    const headers = Object.keys(excelData[0] || {});
    const templateColumns = STUDENT_TEMPLATE.columns;
    console.log(templateColumns);
    
    // 1. Column Name Validation
    const missingColumns = templateColumns.filter(col => !headers.includes(col));
    if (missingColumns.length > 0) {
      errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
    }
    
    // Check for extra columns
    const extraColumns = headers.filter(col => !templateColumns.includes(col));
    if (extraColumns.length > 0) {
      errors.push(`Extra columns found: ${extraColumns.join(', ')}. Please use the template.`);
    }
    
    // 2. Column Sequence Validation
    const isSequenceValid = headers.every((header, index) => 
      index < templateColumns.length ? header === templateColumns[index] : false
    );
    
    if (!isSequenceValid && headers.length === templateColumns.length) {
      errors.push('Column sequence does not match the template. Please maintain the correct column order.');
    }
    
    // 3. Row-wise Validation
    const validatedData = [];
    
    excelData.forEach((row, rowIndex) => {
      const rowErrors = [];
      const rowData = {};
      
      templateColumns.forEach(column => {
        if (!row.hasOwnProperty(column)) {
          rowErrors.push(`Row ${rowIndex + 2}: Missing column '${column}'`);
          return;
        }
        
        const value = row[column];
        const validation = STUDENT_TEMPLATE.validations[column];
        
        if (!validation) return;
        if (column === 'SchoolId' && value !== undefined) {
          rowData[column] = Number(value);
          return;
        };
        
        // Check required fields
        if (validation.required && (value === undefined || value === null || value === '')) {
          rowErrors.push(`Row ${rowIndex + 2}: '${column}' is required`);
          return;
        }
        
        // Skip further validation if value is empty for non-required fields
        if (!validation.required && (value === undefined || value === null || value === '')) {
          return;
        }
        
        // Type validation for string
        if (validation.type === 'string' && value) {
          const stringValue = String(value).trim();
          
          // Pattern validation
          if (validation.pattern && !validation.pattern.test(stringValue)) {
            rowErrors.push(`Row ${rowIndex + 2}: '${column}' has invalid format`);
          }
          
          // Max length validation
          if (validation.maxLength && stringValue.length > validation.maxLength) {
            rowErrors.push(`Row ${rowIndex + 2}: '${column}' exceeds maximum length of ${validation.maxLength} characters`);
          }
          
          rowData[column] = stringValue;
        }
        
        // Type validation for date
        if (validation.type === 'date' && value) {
          let dateValue;
          
          // Try to parse the date
          if (typeof value === 'number') {
            // Excel serial date
            dateValue = new Date(Math.round((value - 25569) * 86400 * 1000));
          } else {
            dateValue = new Date(value);
          }
          
          if (isNaN(dateValue.getTime())) {
            rowErrors.push(`Row ${rowIndex + 2}: '${column}' must be a valid date`);
          } else {
            // Check date range
            if (validation.minDate && dateValue < validation.minDate) {
              rowErrors.push(`Row ${rowIndex + 2}: '${column}' must be after ${validation.minDate.toISOString().split('T')[0]}`);
            }
            if (validation.maxDate && dateValue > validation.maxDate) {
              rowErrors.push(`Row ${rowIndex + 2}: '${column}' must be before ${validation.maxDate.toISOString().split('T')[0]}`);
            }
            
            // Format date to YYYY-MM-DD for API
            rowData[column] = dateValue.toISOString().split('T')[0];
          }
        }
        
        // Allowed values validation
        if (validation.allowedValues && value && !validation.allowedValues.includes(String(value).trim())) {
          rowErrors.push(`Row ${rowIndex + 2}: '${column}' must be one of: ${validation.allowedValues.join(', ')}`);
        }
        
        // Pattern validation (general)
        if (validation.pattern && value && validation.type !== 'string') {
          if (!validation.pattern.test(String(value).trim())) {
            rowErrors.push(`Row ${rowIndex + 2}: '${column}' has invalid format`);
          }
        }
        
        // Security validations for email
        if (column.toLowerCase().includes('email') && value) {
          const email = String(value).toLowerCase();
          if (email.includes('<script') || email.includes('javascript:') || email.includes('data:text/html')) {
            rowErrors.push(`Row ${rowIndex + 2}: '${column}' contains potentially dangerous content`);
          }
        }
      });
      
      // If no errors for this row, add to validated data
      if (rowErrors.length === 0) {
        console.log(rowData);
        validatedData.push(rowData);
      } else {
        errors.push(...rowErrors);
      }
    });
    
    // 4. Duplicate validation (based on email and name)
    const uniqueSet = new Set();
    validatedData.forEach((row, index) => {
      const key = `${row['Student Name']}-${row['Parent Email']}`;
      if (uniqueSet.has(key)) {
        errors.push(`Row ${index + 2}: Duplicate student with same name and parent email found`);
      } else {
        uniqueSet.add(key);
      }
    });
    
    return {
      isValid: errors.length === 0,
      errors,
      validatedData: errors.length === 0 ? validatedData : null
    };
  };

  // Download student import template
  const downloadStudentTemplate = () => {
    const exportToExcel = async () => {
      const workbook = new Excel.Workbook();
      const worksheet = workbook.addWorksheet('Student Import Template');
  
      // Define columns based on template
      const columns = STUDENT_TEMPLATE.columns.map(col => ({
        header: col,
        key: col.toLowerCase().replace(/\s+/g, '_'),
        width: 20
      }));
      
      worksheet.columns = columns;
  
      // Add sample data row
      const sampleRow = {
        'student_name': 'John Doe',
        'dob': '2010-05-15',
        'gender': 'male',
        'class': '5',
        'address': '123 Main Street',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'country': 'India',
        'pincode': '400001',
        'grade': 'A',
        'parent_name': 'Jane Doe',
        'parent_email': 'jane.doe@example.com'
      };
      
      worksheet.addRow(sampleRow);
  
      // Style header row
      worksheet.getRow(1).eachCell((cell) => {
        cell.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Arial', size: 12 };
        cell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: '1A47C2' },
        };
        cell.alignment = { horizontal: 'center', vertical: 'middle' };
        cell.border = {
          top: { style: 'thin' },
          left: { style: 'thin' },
          bottom: { style: 'thin' },
          right: { style: 'thin' }
        };
      });
  
      // Style sample row
      worksheet.getRow(2).eachCell((cell) => {
        cell.font = { name: 'Arial', size: 11 };
        cell.alignment = { vertical: 'middle' };
        cell.border = {
          top: { style: 'thin' },
          left: { style: 'thin' },
          bottom: { style: 'thin' },
          right: { style: 'thin' }
        };
        cell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: 'F0F0F0' },
        };
      });
  
      const buffer = await workbook.xlsx.writeBuffer();
      const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      saveAs(blob, `student_import_template_${new Date().toISOString().split('T')[0]}.xlsx`);
    };
    
    exportToExcel();
  };

  // Handle Excel import with validation
  const handleImportExcel = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.xlsx, .xls';
    fileInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      try {
        const reader = new FileReader();
        reader.onload = async (e) => {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const jsonData = XLSX.utils.sheet_to_json(worksheet);
          
          // Validate the data
          const validationResult = validateStudentExcelData(jsonData);
          
          if (!validationResult.isValid) {
            setImportValidationErrors(validationResult.errors);
            setShowImportModal(true);
            return;
          }
          console.log(validationResult.validatedData);
          
          // Transform validated data for API
          const formattedData = validationResult.validatedData.map(row => ({
            address: row['Address'],
            city: row['City'],
            country: row['Country'],
            state: row['State'],
            pin: row['Pincode'],
            classId: row['Class'],
            schoolId: row['SchoolId'],
            DOB: row['DOB'],
            gender: row['Gender'],
            grade: grades.find(grade => grade.gradeLetter === row['Grade'].toUpperCase())?.gradeId || null,
            parentEmail: row['Parent Email'],
            parentName: row['Parent Name'],
            studentName: row['Student Name']
          }));

          console.log(formattedData);

          // Call API to import data
          const apiUrl = 'http://127.0.0.1:8000/students/importExcel';
          const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(formattedData)
          });

          if (response.ok) {
            const result = await response.json();
            setSuccessMessage(`Successfully imported ${formattedData.length} students!`);
            setImportValidationErrors([]);
            
            // Refresh student data
            const studentsResponse = await fetch('http://127.0.0.1:8000/students/');
            const studentData = await studentsResponse.json();
            const filteredStudent = (studentData?.data || []).filter((student) => 
              student.schoolId === (user?.schoolId || 1)
            );
            setStudents(filteredStudent);
            
            setTimeout(() => {
              setSuccessMessage('');
              setShowImportModal(false);
            }, 3000);
          } else {
            throw new Error(`Import failed: ${response.statusText}`);
          }
        };
        reader.readAsArrayBuffer(file);
      } catch (error) {
        setErrorMessage(`Import error: ${error.message}`);
        setTimeout(() => setErrorMessage(''), 5000);
      }
    };
    fileInput.click();
  };


  const [filteredTeachers, setFilteredTeachers] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [teacherSearch, setTeacherSearch] = useState('');
  const wrapperRef = useRef(null);

  // Debounced API search
  const handleSearch = async (query) => {
    if (query.trim() === '') {
      setFilteredTeachers([]);
      setShowDropdown(false);
      return;
    }

    setIsLoading(true);
    try {
      console.log(query);
      console.log(teachers)
      const response = teachers.filter((teacher) => (teacher.teacherName).toLowerCase().includes(query.toLowerCase()));
      console.log(response);
      setFilteredTeachers(response);
      setShowDropdown(true);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  }

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSelect = (teacher) => {
    setTeacherSearch(teacher.teacherName);
    teacherForm.teacherId = String(teacher.teacherId);
    setShowDropdown(false);
  };

  const handleProvisionSubmit = async () => {
    try {
      if (isStudent) {
    
        if (!studentForm.studentId || !studentForm.provision) {
          alert('Please fill in all student fields');
          return;
        }
        studentForm.changedBy = teacher?.teacherId 
        
        const provision = await fetch('http://127.0.0.1:8000/teachers/provision', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(studentForm),
        });
        
        const response = await provision.json();
        if (provision.ok) {
          setSuccessMessage('Provision added successfully');
          setStudentForm({ studentId: '', provision: '', changedBy: null });
        } else {
          setErrorMessage('Provision addition failed');
        }
      } else {
        if (!teacherForm.provision || !teacherForm.offBoardingDate) {
          alert('Please fill in all teacher fields');
          return;
        }

        teacherForm.changedBy = teacher?.teacherId
        
        const provision = await fetch('http://127.0.0.1:8000/teachers/provision', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(teacherForm),
        });
        
        const response = await provision.json();
        if (provision.ok) {
          setSuccessMessage('Provision added successfully');
          setTeacherForm({ teacherId: '', provision: '', changedBy: null, offBoardingDate: null });
        } else {
          setErrorMessage('Provision addition failed');
        }
      }
      
      setTimeout(() => {
        setSuccessMessage('');
        setErrorMessage('');
      }, 3000);
      
      setShowProvisionModal(false);
    } catch (error) {
      setErrorMessage('Error submitting provision');
      console.error('Provision submission error:', error);
    }
  };

  // Pagination calculations
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(5);
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = sortedStudents.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(sortedStudents.length / itemsPerPage);

  // Handle page change
  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  // Handle items per page change
  const handleItemsPerPageChange = (e) => {
    setItemsPerPage(parseInt(e.target.value));
    setCurrentPage(1);
  };

  // Generate page numbers for pagination
  const getPageNumbers = () => {
    const pageNumbers = [];
    const maxPagesToShow = 5;
    
    if (totalPages <= maxPagesToShow) {
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      const startPage = Math.max(2, currentPage - 1);
      const endPage = Math.min(totalPages - 1, currentPage + 1);
      
      pageNumbers.push(1);
      
      if (startPage > 2) {
        pageNumbers.push('...');
      }
      
      for (let i = startPage; i <= endPage; i++) {
        pageNumbers.push(i);
      }
      
      if (endPage < totalPages - 1) {
        pageNumbers.push('...');
      }
      
      pageNumbers.push(totalPages);
    }
    
    return pageNumbers;
  };

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filters, sortConfig]);

  // Messages display effect
  useEffect(() => {
    if (successMessage || errorMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage('');
        setErrorMessage('');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage, errorMessage]);

  if (loading) return <Loader />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Success/Error Messages */}
      {successMessage && (
        <div className="fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded z-50">
          {successMessage}
        </div>
      )}
      {errorMessage && (
        <div className="fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-50">
          {errorMessage}
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Head Teacher Portal</h1>
              <p className="text-sm text-gray-600 mt-1">Student Management System</p>
            </div>
            <div className='flex gap-4'>
              <button
                onClick={() => setShowProvisionModal(true)}
                className="flex items-center gap-2 bg-orange-400 hover:bg-orange-500 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
              >
                <Plus size={20} />
                Teacher/Student status change
              </button>
              <button
                onClick={() => setShowImportModal(true)}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
              >
                <Upload size={20} />
                Import Student Data
              </button>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors duration-200"
              >
                <LogOut size={18} />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* School Info */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome, {teacher?.teacherName || "User"}!</h2>
              <p className="text-gray-600 mb-1 flex items-center gap-2">
                <span className="font-semibold">School:</span> {schools?.schoolName || 'N/A'}
              </p>
              <p className="text-gray-600 text-sm">
                {schools.address || ''}, {schools.city || ''}, {schools.state || ''} - {schools.pin || ''}
              </p>
            </div>
            <div className="text-right">
              <p className="text-gray-600 text-sm">Login Time: {new Date().toLocaleTimeString()}</p>
              <p className="text-gray-600 text-sm">Date: {new Date().toLocaleDateString()}</p>
            </div>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div className='flex gap-4 flex-wrap'>
            <button
              onClick={() => navigate('/map-teacher')}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Link size={20} />
              Teacher Mapping
            </button>
            <button
              onClick={() => navigate('/add-teacher')}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Plus size={20} />
              Add Teacher
            </button>
            <button
              onClick={() => navigate('/add-student')}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Plus size={20} />
              Add Student
            </button>
            <button
              onClick={() => navigate('/summary')}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <FileText size={20} />
              Summary Report
            </button>
          </div>
        </div>

        {/* Search and Filters Section */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4 mb-4">
            {/* Search Bar */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search students by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
            
            {/* Filter Toggle Button */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Filter size={20} />
              Filters {getFilterCount() > 0 && (
                <span className="bg-indigo-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {getFilterCount()}
                </span>
              )}
            </button>
            
            {/* Clear Filters Button */}
            {getFilterCount() > 0 && (
              <button
                onClick={clearFilters}
                className="flex items-center gap-2 bg-red-100 hover:bg-red-200 text-red-700 px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
              >
                <XIcon size={20} />
                Clear Filters
              </button>
            )}
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="border-t border-gray-200 pt-6 mt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Class Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <GraduationCap size={16} className="inline mr-2" />
                    Class
                  </label>
                  <select
                    value={filters.classId}
                    onChange={(e) => setFilters({...filters, classId: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">All Classes</option>
                    {classes.map((classItem) => (
                      <option key={classItem.classId} value={classItem.classId}>
                        {classItem.className}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Gender Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <User size={16} className="inline mr-2" />
                    Gender
                  </label>
                  <select
                    value={filters.gender}
                    onChange={(e) => setFilters({...filters, gender: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">All Genders</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>

                {/* Grade Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Award size={16} className="inline mr-2" />
                    Grade
                  </label>
                  <select
                    value={filters.grade}
                    onChange={(e) => setFilters({...filters, grade: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">All Grades</option>
                    {Array.from(new Set(grades.map(g => g.gradeLetter))).map(grade => (
                      <option key={grade} value={grade}>{grade === 'N' ? 'NG' : grade}</option>
                    ))}
                  </select>
                </div>

                {/* Performance Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <TrendingUp size={16} className="inline mr-2" />
                    Performance
                  </label>
                  <select
                    value={filters.performance}
                    onChange={(e) => setFilters({...filters, performance: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">All Performance</option>
                    {gradesWithPerformance.map((option, index) => (
                      <option key={index} value={option.performance}>
                        {option.performance}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              
              {/* Active Filter Results */}
              {getFilterCount() > 0 && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {filters.classId && (
                    <span className="inline-flex items-center gap-1 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                      Class: {classes.find(c => c.classId === parseInt(filters.classId))?.className}
                      <button onClick={() => setFilters({...filters, classId: ''})} className="ml-1 hover:text-blue-900">
                        <X size={14} />
                      </button>
                    </span>
                  )}
                  {filters.gender && (
                    <span className="inline-flex items-center gap-1 bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
                      Gender: {filters.gender}
                      <button onClick={() => setFilters({...filters, gender: ''})} className="ml-1 hover:text-purple-900">
                        <X size={14} />
                      </button>
                    </span>
                  )}
                  {filters.grade && (
                    <span className="inline-flex items-center gap-1 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                      Grade: {filters.grade}
                      <button onClick={() => setFilters({...filters, grade: ''})} className="ml-1 hover:text-green-900">
                        <X size={14} />
                      </button>
                    </span>
                  )}
                  {filters.performance && (
                    <span className="inline-flex items-center gap-1 bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
                      Performance: {filters.performance}
                      <button onClick={() => setFilters({...filters, performance: ''})} className="ml-1 hover:text-yellow-900">
                        <X size={14} />
                      </button>
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Stats Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600 mb-1">Total Students</div>
                <div className="text-3xl font-bold text-indigo-600">{students.length}</div>
              </div>
              <div className="bg-indigo-100 p-3 rounded-full">
                <User className="text-indigo-600" size={24} />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600 mb-1">Average Score</div>
                <div className="text-3xl font-bold text-green-600">
                  {students.length > 0 
                    ? Math.round(students.reduce((total, student) => total + getTotalScore(student.studentId), 0) / students.length)
                    : 0}%
                </div>
              </div>
              <div className="bg-green-100 p-3 rounded-full">
                <Award className="text-green-600" size={24} />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600 mb-1">Top Performers</div>
                <div className="text-3xl font-bold text-emerald-600">
                  {students.filter(s => getPerformanceStatus(s) === 'Excellent').length}
                </div>
              </div>
              <div className="bg-emerald-100 p-3 rounded-full">
                <TrendingUp className="text-emerald-600" size={24} />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600 mb-1">Need Improvement</div>
                <div className="text-3xl font-bold text-red-600">
                  {students.filter(s => getPerformanceStatus(s) === 'Poor').length}
                </div>
              </div>
              <div className="bg-red-100 p-3 rounded-full">
                <TrendingDown className="text-red-600" size={24} />
              </div>
            </div>
          </div>
        </div>

        {/* Student Details Table */}
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-blue-500">
            <div className="flex justify-between items-center ">
              <h3 className="text-lg font-semibold text-gray-800 text-white">
                Student Directory ({sortedStudents.length} students)
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600 text-white">
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  Excellent
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                  Good
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                  Average
                </span>
              </div>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    <button 
                      onClick={() => handleSort('roll')}
                      className="flex items-center gap-1 hover:text-indigo-600"
                    >
                      Roll No
                      {sortConfig.key === 'roll' && (
                        sortConfig.direction === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    <button 
                      onClick={() => handleSort('name')}
                      className="flex items-center gap-1 hover:text-indigo-600"
                    >
                      Student Details
                      {sortConfig.key === 'name' && (
                        sortConfig.direction === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    <button 
                      onClick={() => handleSort('class')}
                      className="flex items-center gap-1 hover:text-indigo-600"
                    >
                      Academic Info
                      {sortConfig.key === 'class' && (
                        sortConfig.direction === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    <button 
                      onClick={() => handleSort('score')}
                      className="flex items-center gap-1 hover:text-indigo-600"
                    >
                      Performance Metrics
                      {sortConfig.key === 'score' && (
                        sortConfig.direction === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {currentItems.map((student) => {
                  const totalScore = getTotalScore(student.studentId);
                  const performance = getPerformanceStatus(totalScore);
                  const performanceColor = getPerformanceColor(performance);
                  const subjectScores = getSubjectScores(student.studentId);
                  
                  return (
                    <tr key={student.studentId} className="hover:bg-gray-50 transition-colors">
                      {/* Roll Number */}
                      <td className="px-6 py-4">
                        <div className="text-sm font-medium text-gray-900">
                          {student.rollId || 'N/A'}
                        </div>
                      </td>
                      
                      {/* Student Details */}
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center">
                            <User className="text-indigo-600" size={20} />
                          </div>
                          <div>
                            <a 
                              href="" 
                              onClick={(e) => {
                                e.preventDefault();
                                navigate(`/final-report/${student.studentId}`);
                              }}
                              className="text-sm font-semibold text-gray-900 hover:text-indigo-600 transition-colors"
                            >
                              {student.studentName}
                            </a>
                            <div className="flex items-center gap-2 mt-1">
                              <span className="text-xs text-gray-600">{student.gender || 'N/A'}</span>
                              <span className="text-xs text-gray-400">•</span>
                              <span className="text-xs text-gray-600">
                                {student.DOB ? new Date(student.DOB).toLocaleDateString('en-US', { 
                                  month: 'short', 
                                  day: 'numeric', 
                                  year: 'numeric' 
                                }) : 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </td>
                      
                      {/* Academic Info */}
                      <td className="px-6 py-4">
                        <div className="space-y-1">
                          <div className="text-sm font-medium text-gray-900">
                            {classes.find(c => c.classId === student.classId)?.className || 'N/A'}
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {subjectScores.slice(0, 3).map((subject, idx) => (
                              <span 
                                key={idx} 
                                className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded"
                                title={`${subject.subject}: ${subject.score}/${subject.maxScore}`}
                              >
                                {subject.subject.substring(0, 3)}
                              </span>
                            ))}
                            {subjectScores.length > 3 && (
                              <span className="text-xs text-gray-400">
                                +{subjectScores.length - 3} more
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      
                      {/* Performance Metrics */}
                      <td className="px-6 py-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Total Score</span>
                            <span className="text-sm font-semibold">{totalScore}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Grade</span>
                            <span className="text-sm font-medium px-2 py-1 rounded bg-blue-100 text-blue-800">
                              {calculatePerformance(totalScore) === 'N' ? 'NG': calculatePerformance(totalScore)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Performance</span>
                            <span className={`text-xs font-medium px-2 py-1 rounded ${performanceColor}`}>
                              {performance}
                            </span>
                          </div>
                        </div>
                      </td>
                      
                      {/* Actions */}
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <button 
                            onClick={() => navigate(`/final-report/${student.studentId}`)}
                            className="p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
                            title="View Full Report"
                          >
                            <FileText size={18} />
                          </button>
                          <button 
                            onClick={() => navigate(`/edit-student/${student.studentId}`)}
                            className="p-2 text-green-600 hover:text-green-800 hover:bg-green-50 rounded-lg transition-colors"
                            title="Edit Student"
                          >
                            <Edit size={18} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            
            {sortedStudents.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <div className="mb-4">
                  <Search size={48} className="mx-auto text-gray-300" />
                </div>
                <p className="text-lg font-medium text-gray-700 mb-2">No students found</p>
                <p className="text-gray-600">
                  {getFilterCount() > 0 
                    ? "Try adjusting your filters or search term"
                    : "No students available in the system"}
                </p>
              </div>
            )}
          </div>
          
          {/* Table Footer */}
          {currentItems.length > 0 && (
            <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600">
                  Showing <span className="font-semibold">{Math.min(currentItems.length, 25)}</span> of{' '}
                  <span className="font-semibold">{sortedStudents.length}</span> students
                </div>
                <div className="flex items-center gap-4">
                  <button 
                    onClick={handleExportExcel}
                    className="flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                  >
                    <Download size={16} />
                    Export Results
                  </button>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Sort by:</span>
                    <select 
                      value={sortConfig.key}
                      onChange={(e) => handleSort(e.target.value)}
                      className="text-sm border border-gray-300 rounded px-2 py-1"
                    >
                      <option value="name">Name</option>
                      <option value="class">Class</option>
                      <option value="score">Score</option>
                      <option value="grade">Grade</option>
                    </select>
                    <button
                      onClick={() => setSortConfig(prev => ({ ...prev, direction: prev.direction === 'asc' ? 'desc' : 'asc' }))}
                      className="text-gray-600 hover:text-gray-800"
                    >
                      {sortConfig.direction === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Selected Report Modal */}
      {selectedReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            <div className="bg-indigo-600 text-white p-4 flex items-center justify-between">
              <h3 className="text-xl font-bold">{selectedReport.subject} Report</h3>
              <button
                onClick={() => setSelectedReport(null)}
                className="text-white hover:text-gray-200"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
              <div className="mb-4 flex items-center gap-4">
                <button
                  onClick={() => setZoomLevel(Math.max(50, zoomLevel - 10))}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                >
                  <ZoomOut size={18} />
                  Zoom Out
                </button>
                <span className="font-medium">{zoomLevel}%</span>
                <button
                  onClick={() => setZoomLevel(Math.min(200, zoomLevel + 10))}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                >
                  <ZoomIn size={18} />
                  Zoom In
                </button>
              </div>

              <div 
                className="border-2 border-gray-300 rounded-lg p-8 bg-gray-50 mb-6 transition-transform"
                style={{ transform: `scale(${zoomLevel / 100})`, transformOrigin: 'top left' }}
              >
                <div className="bg-white p-6 rounded shadow">
                  <h4 className="text-2xl font-bold mb-4">{selectedReport.subject} Report</h4>
                  <p className="text-gray-700 mb-2"><strong>Teacher:</strong> {selectedReport.teacher}</p>
                  <p className="text-gray-700 mb-4"><strong>Submitted:</strong> {selectedReport.sentDate}</p>
                  <div className="border-t pt-4">
                    <p className="text-gray-600 leading-relaxed">
                      This is a sample report document. In a real application, this would display the actual report content,
                      including grades, assessments, observations, and detailed student performance data.
                      The report includes comprehensive analysis of student progress throughout the term.
                    </p>
                  </div>
                </div>
              </div>

              {selectedReport.status === 'pending' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Comments (Optional for Approval, Required for Rejection)
                    </label>
                    <textarea
                      value={comment}
                      onChange={(e) => setComment(e.target.value)}
                      placeholder="Add your comments here..."
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      rows="4"
                    />
                  </div>

                  <div className="flex gap-4">
                    <button
                      onClick={handleApprove}
                      className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
                    >
                      <CheckCircle size={20} />
                      Approve Report
                    </button>
                    <button
                      onClick={handleReject}
                      className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
                    >
                      <XCircle size={20} />
                      Reject Report
                    </button>
                  </div>
                </div>
              )}

              {selectedReport.status !== 'pending' && (
                <div className={`p-4 rounded-lg ${selectedReport.status === 'approved' ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                  <p className="font-semibold mb-2">
                    Status: {selectedReport.status.charAt(0).toUpperCase() + selectedReport.status.slice(1)}
                  </p>
                  {selectedReport.comments && (
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-1">Head Teacher Comments:</p>
                      <p className="text-gray-700 italic">{selectedReport.comments}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Pagination Footer */}
      {sortedStudents.length > 0 && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">Show:</span>
                <select
                  value={itemsPerPage}
                  onChange={handleItemsPerPageChange}
                  className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
                >
                  <option value="5">5</option>
                  <option value="10">10</option>
                  <option value="25">25</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </select>
                <span className="text-sm text-gray-600">per page</span>
              </div>
              <div className="text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className={`px-3 py-1 rounded border ${currentPage === 1 ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
              >
                <ChevronLeft size={16} />
              </button>
              
              {getPageNumbers().map((page, index) => (
                <button
                  key={index}
                  onClick={() => typeof page === 'number' ? handlePageChange(page) : null}
                  className={`px-3 py-1 rounded border ${currentPage === page ? 'bg-indigo-600 text-white border-indigo-600' : 'text-gray-700 border-gray-300 hover:bg-gray-100'} ${typeof page !== 'number' ? 'cursor-default hover:bg-transparent' : ''}`}
                  disabled={typeof page !== 'number'}
                >
                  {page}
                </button>
              ))}
              
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className={`px-3 py-1 rounded border ${currentPage === totalPages ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
            <div className="bg-green-600 text-white p-4 flex items-center justify-between">
              <h2 className="text-xl font-bold">Import Students</h2>
              <button
                onClick={() => {
                  setShowImportModal(false);
                  setImportValidationErrors([]);
                }}
                className="text-white hover:text-gray-200 text-2xl"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
              <div className="space-y-6">
                {/* Download Template Section */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-2">Step 1: Download Template</h3>
                  <p className="text-sm text-blue-700 mb-3">
                    Download the Excel template with the correct format and sample data.
                  </p>
                  <button
                    onClick={downloadStudentTemplate}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    <Download size={18} />
                    Download Student Import Template
                  </button>
                </div>
                
                {/* Import Excel Section */}
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-green-800 mb-2">Step 2: Import Excel File</h3>
                  <p className="text-sm text-green-700 mb-3">
                    Fill the template with your student data and upload it here.
                  </p>
                  <button
                    onClick={handleImportExcel}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    <Upload size={18} />
                    Upload Excel File
                  </button>
                </div>
                
                {/* Validation Errors */}
                {importValidationErrors.length > 0 && (
                  <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                    <h3 className="font-semibold text-red-800 mb-2">Validation Errors Found</h3>
                    <p className="text-sm text-red-700 mb-3">
                      Please fix the following errors in your Excel file:
                    </p>
                    <div className="max-h-60 overflow-y-auto">
                      <ul className="space-y-1">
                        {importValidationErrors.slice(0, 10).map((error, index) => (
                          <li key={index} className="text-sm text-red-700 flex items-start gap-2">
                            <span className="mt-1">•</span>
                            <span>{error}</span>
                          </li>
                        ))}
                        {importValidationErrors.length > 10 && (
                          <li className="text-sm text-red-700">
                            ... and {importValidationErrors.length - 10} more errors
                          </li>
                        )}
                      </ul>
                    </div>
                  </div>
                )}
                
                {/* Instructions */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-2">Important Instructions</h3>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li>• Do not change the column names or sequence</li>
                    <li>• Use the exact format as shown in the template</li>
                    <li>• All required fields must be filled</li>
                    <li>• Dates should be in YYYY-MM-DD format</li>
                    <li>• Grade must be A-F or N (for Not Graded)</li>
                    <li>• Parent email must be valid</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Provision Modal */}
      {showProvisionModal && (
        <div className="w-full fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full">
            <div className="flex justify-between items-center p-6 border-b border-gray-200">
              <h2 className="text-2xl font-bold text-gray-800">Inactive (Student/Teacher) Provision</h2>
              <button
                onClick={() => setShowProvisionModal(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X size={24} />
              </button>
            </div>

            {/* Toggle Switch */}
            <div className="px-6 pt-6">
              <div className="flex items-center justify-center gap-4 mb-6">
                <span className={`text-sm font-medium transition-colors ${isStudent ? 'text-indigo-600' : 'text-gray-500'}`}>
                  Student
                </span>
                <button
                  role="switch"
                  aria-checked={!isStudent}
                  onClick={() => setIsStudent(!isStudent)}
                  className={`relative inline-flex h-7 w-14 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2
                    ${!isStudent ? 'bg-indigo-600' : 'bg-gray-300'}`}
                >
                  <span
                    aria-hidden="true"
                    className={`pointer-events-none inline-block h-6 w-6 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out
                      ${!isStudent ? 'translate-x-7' : 'translate-x-0'}`}
                  ></span>
                </button>
                <span className={`text-sm font-medium transition-colors ${!isStudent ? 'text-indigo-600' : 'text-gray-500'}`}>
                  Teacher
                </span>
              </div>
            </div>
            
            <div className="p-6 pt-0">
              {/* Student Form */}
              {isStudent ? (
                <div className="space-y-4">
                  {/* select student */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Student Name
                    </label>
                    <select
                      value={studentForm.studentId}
                      onChange={(e) => setStudentForm({...studentForm, studentId: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 bg-white rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    >
                      <option value="" disabled>Select a student</option>
                      {students.map((student) => (
                        <option key={student.studentId} value={student.studentId}>
                          {student.studentName} (class-{student.classId}) ({student.rollId})
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Parent Email
                    </label>
                    <input
                      type="email"
                      value={studentForm.email}
                      onChange={(e) => setStudentForm({...studentForm, email: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      placeholder="e.g., student@gmail.com"
                    />
                  </div> */}
                  
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Class
                    </label>
                    <select
                      value={studentForm.classId}
                      onChange={(e) => setStudentForm({...studentForm, classId: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 bg-white rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    >
                      <option value="" disabled>Select a class</option>
                      {classes.map((classItem) => (
                        <option key={classItem.classId} value={classItem.classId}>
                          {classItem.className}
                        </option>
                      ))}
                    </select>
                  </div> */}
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Inactive Details
                    </label>
                    <textarea
                      value={studentForm.provision}
                      onChange={(e) => setStudentForm({...studentForm, provision: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 h-24 resize-none"
                      placeholder="Enter provision details..."
                    />
                  </div>
                </div>
              ) : (
                /* Teacher Form */
                <div className="space-y-4">
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Teacher Name
                    </label>
                    <input
                      type="text"
                      value={teacherForm.teacherName}
                      onChange={(e) => setTeacherForm({...teacherForm, teacherName: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      placeholder="Enter teacher name"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email
                    </label>
                    <input
                      type="email"
                      value={teacherForm.email}
                      onChange={(e) => setTeacherForm({...teacherForm, email: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      placeholder="e.g., teacher@gmail.com"
                    />
                  </div> */}

                  {/* implement type search */}
                  <div ref={wrapperRef} className="relative">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Teacher Name
                    </label>
                    
                    <div className="relative">
                      <input
                        type="text"
                        value={teacherSearch}
                        onChange={(e) => handleSearch(e.target.value)}
                        onFocus={() => teacherSearch.trim() !== '' && setShowDropdown(true)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                        placeholder="Start typing to search..."
                      />
                      
                      {/* Loading indicator */}
                      {isLoading && (
                        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-600"></div>
                        </div>
                      )}
                    </div>

                    {/* Dropdown Results */}
                    {showDropdown && (
                      <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-xl">
                        {isLoading ? (
                          <div className="p-4 text-center">
                            <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div>
                            <p className="text-sm text-gray-500 mt-2">Searching...</p>
                          </div>
                        ) : filteredTeachers.length > 0 ? (
                          <div className="max-h-60 overflow-y-auto">
                            {filteredTeachers.map((teacher) => (
                              <div
                                key={teacher.teacherId}
                                onClick={() => handleSelect(teacher)}
                                className="p-3 hover:bg-indigo-50 cursor-pointer border-b last:border-b-0 group"
                              >
                                <div className="flex items-center justify-between">
                                  <div>
                                    <p className="font-medium text-gray-900 group-hover:text-indigo-700">
                                      {teacher.teacherName}
                                    </p>
                                    <div className="flex items-center gap-2 mt-1">
                                      {teacher.teacherEmail && (
                                        <span className="text-xs text-gray-500">{teacher.teacherEmail}</span>
                                      )}
                                    </div>
                                  </div>
                                  <svg className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                                  </svg>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : teacherSearch.trim() !== '' ? (
                          <div className="p-4 text-center text-gray-500">
                            No teachers found for "{teacherSearch}"
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>
                  
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Subject
                    </label>
                    <select
                      value={teacherForm.subject}
                      onChange={(e) => setTeacherForm({...teacherForm, subject: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 bg-white rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    >
                      <option value="" disabled>Select a subject</option>
                      {subjects.map((subject) => (
                        <option key={subject.subjectId} value={subject.subjectId}>
                          {subject.subjectName}
                        </option>
                      ))}
                    </select>
                  </div> */}
                  
                  {/* <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Class
                    </label>
                    <select
                      value={teacherForm.classId}
                      onChange={(e) => setTeacherForm({...teacherForm, classId: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 bg-white rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    >
                      <option value="" disabled>Select a class</option>
                      {classes.map((classItem) => (
                        <option key={classItem.classId} value={classItem.classId}>
                          {classItem.className}
                        </option>
                      ))}
                    </select>
                  </div> */}

                  {/* offBoarding date if user want to offboard */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Offboarding Date
                    </label>
                    <input
                      type="date"
                      value={teacherForm.offBoardingDate || ''}
                      onChange={(e) => setTeacherForm({...teacherForm, offBoardingDate: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Inactive Details
                    </label>
                    <textarea
                      value={teacherForm.provision}
                      onChange={(e) => setTeacherForm({...teacherForm, provision: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 h-24 resize-none"
                      placeholder="Enter provision details..."
                    />
                  </div>
                </div>
              )}
              
              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleProvisionSubmit}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg transition-colors duration-200 font-medium"
                >
                  Submit
                </button>
                <button
                  onClick={() => setShowProvisionModal(false)}
                  className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 rounded-lg transition-colors duration-200 font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeadTeacher;