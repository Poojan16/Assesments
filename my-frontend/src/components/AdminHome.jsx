import React, { useState, useMemo, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { School, Users, GraduationCap, TrendingUp, Filter, Search, X, ChevronLeft, ChevronRight, Download, AlertCircle, Plus } from 'lucide-react';
import Excel from 'exceljs';
import { saveAs } from 'file-saver';
import { useNavigate } from 'react-router-dom';
import { Upload, LogOut } from "lucide-react";
import Loader from './loader';
import * as XLSX from 'xlsx'; // Import the xlsx library
import Select from 'react-select';
import { all } from 'axios';
import { initializeAuth, logout } from '../authSlice';
import { useSelector, useDispatch } from 'react-redux';
import EducationDashboard from './admin_dashboard_prac';


const AdminDashboard2 = () => {
  const [selectedSchool, setSelectedSchool] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [discontinueModal, setDiscontinueModal] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(8);
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState('asc');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPerformance, setFilterPerformance] = useState('all');
  const [filterCity, setFilterCity] = useState('all');
  const [showFilters, setShowFilters] = useState(false);
  const [teacherPage, setTeacherPage] = useState(1);
  const [studentPage, setStudentPage] = useState(1);
  const [teacherSearch, setTeacherSearch] = useState('');
  const [studentSearch, setStudentSearch] = useState('');
  const [filterState, setFilterState] = useState('all');
  const [filterPincode, setFilterPincode] = useState('all');
  const [loading,setLoading] = useState(true);
  const [schools, setSchools] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [students, setStudents] = useState([]);
  const [error, setError] = useState(null);
  const [roles, setRoles] = useState([]);
  const [classes, setClasses] = useState([]);
  const [grades, setGrades] = useState([]);
  const [allTeachers, setAllTeachers] = useState([]);
  const [allStudents, setAllStudents] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [popUpMsg, setPopupMsg] = useState('');
  const navigate = useNavigate();
  const [importModal, setImportModal] = useState(false);
  const [importValidationErrors, setImportValidationErrors] = useState({});
  const [scores, setScores] = useState([]);

  const {user} = useSelector((state) => state.auth);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(initializeAuth());
  }, [dispatch]);

  useEffect(() => {
    const generateData = async () => {
      setLoading(true);
      try {
        const responses = await Promise.all([
          fetch('http://127.0.0.1:8000/admin/schools/'),
          fetch('http://127.0.0.1:8000/teachers/'),
          fetch('http://127.0.0.1:8000/students/'),
          fetch('http://127.0.0.1:8000/roles/'),
          fetch('http://127.0.0.1:8000/classes/'),
          fetch('http://127.0.0.1:8000/grades/'),
          fetch('http://127.0.0.1:8000/subjects/'),
          fetch('http://127.0.0.1:8000/students/score')
        ]);
  
        responses.forEach(res => {
          if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        });
  
        const [
          schoolRes,
          teacherRes,
          studentRes,
          roleRes,
          classRes,
          gradeRes,
          subjectRes,
          scoreRes
        ] = await Promise.all(responses.map(r => r.json()));
  
        const schools = schoolRes.data || [];
        const teachers = teacherRes.data || [];
        const students = studentRes.data || [];
        const roles = roleRes.data || [];
        const classes = classRes.data || [];
        const grades = gradeRes.data || [];
        const subjects = subjectRes.data || [];
        const scores = scoreRes.data || [];
  
        setAllTeachers(teachers);
        setAllStudents(students);
        setRoles(roles);
        setClasses(classes);
        setGrades(grades);
        setSubjects(subjects);
        setScores(scores);
  
        const AvgScoreSchoolWise = schools.map(school => {
          const schoolStudents = students.filter(
            student => student.schoolId === school.schoolId
          );
  
          const studentAverages = schoolStudents.map(student => {
            const studentScores = scores.filter(
              score => score.studentId === student.studentId
            );
  
            if (studentScores.length === 0) return 0;
  
            const total = studentScores.reduce(
              (sum, s) => sum + s.score,
              0
            );
  
            return total / studentScores.length;
          });
  
          const schoolAvg =
            studentAverages.length > 0
              ? studentAverages.reduce((sum, avg) => sum + avg, 0) /
                studentAverages.length
              : 0;
  
          return {
            ...school,
            avgScore: schoolAvg
          };
        });
  
        console.log(AvgScoreSchoolWise);
        setSchools(AvgScoreSchoolWise);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };
  
    generateData();
  }, []);

  // Filtering and sorting logic
  const filteredAndSortedSchools = useMemo(() => {
    let filtered = schools.filter(school => {
      const matchesSearch = school.schoolName.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = filterStatus === 'all' || school.status === filterStatus;
      const matchesPerformance = filterPerformance === 'all' ||
        (filterPerformance === 'high' && school.avgScore >= 80) ||
        (filterPerformance === 'medium' && school.avgScore >= 70 && school.avgScore < 80) ||
        (filterPerformance === 'low' && school.avgScore < 70);

      console.log(filterCity, filterState, filterPincode);

      const matchesCity = filterCity === 'all' || school.city === filterCity;
      const matchesState = filterState === 'all' || school.state === filterState;
      const matchesZip = filterPincode === 'all' || school.pin === filterPincode;

      return matchesSearch && matchesStatus && matchesPerformance && matchesCity && matchesState && matchesZip;
      
    });

    filtered.sort((a, b) => {
      if(sortOrder === 'asc') {
        return a['schoolId'] > b['schoolId'] ? 1 : -1;
      } else {
        return a['schoolId'] < b['schoolId'] ? 1 : -1;
      }
    });

    return filtered;
  }, [schools, searchTerm, sortBy, sortOrder, filterStatus, filterPerformance, filterCity, filterState, filterPincode]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedSchools.length / itemsPerPage);
  const paginatedSchools = filteredAndSortedSchools.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  console.log(filteredAndSortedSchools);


  const handleSchoolClick = (school) => {
    setSelectedSchool(school);
    setActiveTab('teachers');
    setTeacherPage(1);
    setStudentPage(1);
  };

  const handleDiscontinue = (type, item) => {
    setDiscontinueModal({ type, item });
  };

  const confirmDiscontinue = () => {
    alert(`${discontinueModal.type} "${discontinueModal.item.name}" has been discontinued`);
    setDiscontinueModal(null);
  };

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      dispatch(logout());
      navigate('/');
    }
  };
    
  const handleImportExcel = () => {
    const file = document.createElement('input');
    file.type = 'file';
    file.accept = '.xlsx, .xls';
    file.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0]; // Get the first sheet
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet); // Convert to JSON
        handleImport(jsonData);
      };
      reader.readAsArrayBuffer(file);
    };
    file.click();
  };

  // Import Excel Validations and implementations

  const Modal = ({ title, children, onClose, width = 'max-w-lg' }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className={`bg-white rounded-2xl shadow-2xl ${width} w-full max-h-[90vh] overflow-hidden`}>
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold">{title}</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200 text-2xl"
          >
            ✕
          </button>
        </div>
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
          {children}
        </div>
      </div>
    </div>
  );
  
const REQUIRED_TEMPLATE = {
  columns: [
    'School Name',
    'School Email',
    'Primary Contact No',
    'Secondary Contact No',
    'Additional Contact No',
    'Established Year',
    'Board',
    'Address',
    'City',
    'State',
    'Country',  
    'Pincode',
    'Max Class Limit',
    'Students Per Class'
  ],
  validations: {
    'Address': { type: 'string', maxLength: 500, required: true },
    'City': { type: 'string', pattern: /^[A-Za-z\s]+$/, required: true },
    'State': { type: 'string', pattern: /^[A-Za-z\s]+$/, required: true },
    'Country': { type: 'string', pattern: /^[A-Za-z\s]+$/, required: true, allowedValues: ['India', 'USA', 'UK', 'Canada', 'Australia'] }, // Customize as needed
    'Pincode': { type: 'string', pattern: /^\d{5,6}$/, required: true },
    'School Name': { type: 'string', maxLength: 200, required: true },
    'Established Year': { 
      type: 'number', 
      min: 1800, 
      max: new Date().getFullYear(),
      required: true 
    },
    'Board': { 
      type: 'string', 
      required: true 
    },
    'Primary Contact No': { 
      type: 'string', 
      pattern: /^[\d\s\+\-\(\)]{10,15}$/, 
      required: true 
    },
    'Secondary Contact No': { 
      type: 'string', 
      pattern: /^[\d\s\+\-\(\)]{10,15}$/, 
      required: false 
    },
    'Additional Contact No': { 
      type: 'string', 
      pattern: /^[\d\s\+\-\(\)]{10,15}$/, 
      required: false 
    },
    'School Email': { 
      type: 'string', 
      pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      required: true 
    },
    'Max Class Limit': { 
      type: 'number', 
      min: 1, 
      max: 12, 
      required: true 
    },
    'Students Per Class': { 
      type: 'number', 
      min: 1, 
      max: 30, 
      required: true 
    }
  }
};

const validateExcelData = (excelData) => {
  const errors = [];
  
  if (!Array.isArray(excelData) || excelData.length === 0) {
    errors.push('No data found in the Excel file');
    return { isValid: false, errors };
  }

  // Get headers from first row
  const headers = Object.keys(excelData[0] || {});
  
  // 1. Column Name Validation
  const templateColumns = REQUIRED_TEMPLATE.columns;
  
  // Check if all required columns are present
  const missingColumns = templateColumns.filter(col => !headers.includes(col));
  if (missingColumns.length > 0) {
    errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
  }
  
  // Check for extra columns
  const extraColumns = headers.filter(col => !templateColumns.includes(col));
  if (extraColumns.length > 0) {
    errors.push(`Extra columns found: ${extraColumns.join(', ')}`);
  }
  
  // 2. Column Sequence Validation
  const isSequenceValid = headers.every((header, index) => 
    index < templateColumns.length ? header === templateColumns[index] : false
  );
  
  if (!isSequenceValid && headers.length === templateColumns.length) {
    errors.push('Column sequence does not match the template. Please maintain the correct column order.');
  }
  
  // 3. Row-wise Validation
  excelData.forEach((row, rowIndex) => {
    templateColumns.forEach(column => {
      // if school name length require at least > 6
      if (column === 'School Name' && row[column] && row[column].length < 6) {
        errors.push(`Row ${rowIndex + 2}: School Name must be at least 6 characters long`);
      }

      // email should be unique in all rows
      if (column === 'School Email') {
        const email = row[column];
        if (email && excelData.some((r, i) => i !== rowIndex && r[column] === email)) {
          errors.push(`Row ${rowIndex + 2}: ${column} should be unique`);
        }
      }

      // check secondary and additional conact are optional, all should be unique
      if (column === 'Secondary Contact No' || column === 'Additional Contact No') {
        if (row['Primary Contact No'] === row[column]) {
          errors.push(`Row ${rowIndex + 2}: ${column} should not be same as Primary Contact No`);
        }
      }
      
      if (!row.hasOwnProperty(column)) {
        errors.push(`Row ${rowIndex + 2}: Missing column '${column}'`);
        return;
      }
      
      const value = row[column];
      const validation = REQUIRED_TEMPLATE.validations[column];
      
      if (!validation) return;
      
      // Check required fields
      if (validation.required && (value === undefined || value === null || value === '')) {
        errors.push(`Row ${rowIndex + 2}: '${column}' is required`);
        return;
      }
      
      // Skip further validation if value is empty for non-required fields
      if (!validation.required && (value === undefined || value === null || value === '')) {
        return;
      }
      
      // Type validation
      if (validation.type === 'number') {
        const numValue = Number(value);
        if (isNaN(numValue)) {
          errors.push(`Row ${rowIndex + 2}: '${column}' must be a number`);
        } else {
          if (validation.min !== undefined && numValue < validation.min) {
            errors.push(`Row ${rowIndex + 2}: '${column}' must be at least ${validation.min}`);
          }
          if (validation.max !== undefined && numValue > validation.max) {
            errors.push(`Row ${rowIndex + 2}: '${column}' must be at most ${validation.max}`);
          }
        }
      }
      
      // Pattern validation
      if (validation.pattern && value !== undefined && value !== null && value !== '') {
        const stringValue = String(value).trim();
        if (!validation.pattern.test(stringValue)) {
          errors.push(`Row ${rowIndex + 2}: '${column}' has invalid format`);
        }
      }
      
      // Allowed values validation
      if (validation.allowedValues && !validation.allowedValues.includes(value)) {
        errors.push(`Row ${rowIndex + 2}: '${column}' must be one of: ${validation.allowedValues.join(', ')}`);
      }
      
      // Max length validation
      if (validation.maxLength && String(value).length > validation.maxLength) {
        errors.push(`Row ${rowIndex + 2}: '${column}' exceeds maximum length of ${validation.maxLength} characters`);
      }
      
      // Security validations
      if (column.toLowerCase().includes('email')) {
        // Prevent email injection
        const email = String(value).toLowerCase();
        if (email.includes('<script') || email.includes('javascript:') || email.includes('data:text/html')) {
          errors.push(`Row ${rowIndex + 2}: '${column}' contains potentially dangerous content`);
        }
      }
      
      if (column.toLowerCase().includes('contact')) {
        // Sanitize phone numbers
        const phone = String(value).replace(/[^\d\+]/g, '');
        if (phone.length < 10 || phone.length > 10) {
          errors.push(`Row ${rowIndex + 2}: '${column}' must be 10 digits`);
        }
      }
      
      // Cross-column validation (example)
      if (column === 'Students Per Class' && row['Max Class Limit']) {
        const studentsPerClass = Number(row['Students Per Class']);
        const maxClassLimit = Number(row['Max Class Limit']);
        if (!isNaN(studentsPerClass) && !isNaN(maxClassLimit)) {
          const totalCapacity = studentsPerClass * maxClassLimit;
          if (totalCapacity > 5000) { // Example security limit
            errors.push(`Row ${rowIndex + 2}: Total school capacity exceeds security limit`);
          }
        }
      }
    });
  });
  
  // 4. Duplicate row validation
  const rowHashes = new Set();
  excelData.forEach((row, rowIndex) => {
    const rowString = templateColumns.map(col => row[col]).join('|');
    if (rowHashes.has(rowString)) {
      errors.push(`Row ${rowIndex + 2}: Duplicate data found`);
    }
    rowHashes.add(rowString);
  });
  
  return {
    isValid: errors.length === 0,
    errors,
    validatedData: errors.length === 0 ? excelData : null
  };
};
const handleImport = (excelData) => {

  // First validate the data
  const validationResult = validateExcelData(excelData);
  
  if (!validationResult.isValid) {
    // Show errors to user
    // const errorMessage = validationResult.errors.join('\n');
    const errorMessage = validationResult.errors.map(error => `- ${error}`).join('\n');
    alert(`Validation Errors:\n\n${errorMessage}\n\nPlease correct the Excel file and try again.`);
    // const timer = setTimeout(() => {
    //   setPopupMsg('');
    //   clearTimeout(timer);
    // }, 10000);
    
    // Optional: Log errors to console for debugging
    console.error('Excel Validation Errors:', validationResult.errors);
    return;
  }
  
  const url = 'http://127.0.0.1:8000/importExcel';
  const method = 'POST';

  const Data = [];

  // Transform validated data for API
  for (const item of validationResult.validatedData) {
    const data = {
      'address': item.Address,
      'city': item.City,
      'state': item.State,
      'country': item.Country, // Added country
      'pin': item.Pincode,
      'schoolName': item['School Name'],
      'established_year': item['Established Year'],
      'board': item.Board,
      'primaryContactNo': item['Primary Contact No'],
      'secondaryContactNo': item['Secondary Contact No'],
      'additionalContactNo': item['Additional Contact No'],
      'schoolEmail': item['School Email'],
      'maxClassLimit': item['Max Class Limit'],
      'studentsPerClass': item['Students Per Class'],
    };
    Data.push(data);
  }

  // Optional: Add CSRF token if needed
  const headers = {
    'Content-Type': 'application/json',
  };

  fetch(url, {
    method,
    headers,
    body: JSON.stringify(Data),
  })
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    console.log('Import successful:', data);
    setPopupMsg('Data Imported Successfully');
    const timer = setTimeout(() => {
      setPopupMsg('');
      setImportModal(false);
      clearTimeout(timer);
      window.location.reload();
    }, 3000);
  })
  .catch((error) => {
    console.error('Error:', error);
    setPopupMsg('Error in importing data');
    const timer = setTimeout(() => {
      setPopupMsg('');
      setImportModal(false);
      clearTimeout(timer);
    })
  }, 3000);
};

  const exportData = () => {
    const exportToExcel = async (data, fileName) => {
          const workbook = new Excel.Workbook();
          const worksheet = workbook.addWorksheet('Sheet1');

 


        
          // Define columns and headers based on school data
          worksheet.columns = [
            { header: 'School Name', key: 'schoolName'},
            { header: 'Established Year', key: 'established_year'},
            { header: 'Board', key: 'board'},
            { header: 'Address', key: 'address'},
            { header: 'Primary ContactNo', key: 'primaryContactNo'},
            { header: 'Secondary ContactNo', key: 'secondaryContactNo'},
            { header: 'Additional ContactNo', key: 'additionalContactNo'},
            { header: 'School Email', key: 'schoolEmail'},
            { header: 'City', key: 'city'},
            { header: 'State', key: 'state'},
            { header: 'Pincode', key: 'pin'},
            { header: 'Class Limit', key: 'maxClassLimit'},
            { header: 'Student Per Class', key: 'studentsPerClass'},
            // Add other columns
          ];
    
          console.log(data[0])
    
          const rowsToAdd = data.map(row => ({
              ...row, // Spread existing data properties
              city: row.city,
              state: row.state,
              pin: row.pin,
              established: new Date(row.established).getFullYear(),
          }));

          console.log(rowsToAdd)
        
          // Add data (which is already processed for capitalization)
          worksheet.addRows(rowsToAdd);
        
          // Apply styling to the header row
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
        
          // Apply styling to data rows
          worksheet.eachRow({ includeEmpty: false }, (row, rowNumber) => {
            if (rowNumber > 1) { // Skip header row
              row.eachCell((cell) => {
                cell.font = { name: 'Arial', size: 11 }; // Black font
                cell.border = {
                    top: {style:'thin'},
                    left: {style:'thin'},
                    bottom: {style:'thin'},
                    right: {style:'thin'}
                };
                cell.alignment = { vertical: 'middle', horizontal: 'left' };
              });
            }
          });
        
          // Generate the Excel file buffer and save it
          const buffer = await workbook.xlsx.writeBuffer();
          const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
          saveAs(blob, `${fileName}.xlsx`);
        };
        exportToExcel(schools,"schools");
  };

  const downloadTemplate = () => {
      const exportToExcel = async (data, fileName) => {
        const workbook = new Excel.Workbook();
        const worksheet = workbook.addWorksheet('Sheet1');
  
        // Define columns and headers
        worksheet.columns = [
          { header: 'School Name', key: 'schoolName'},
          { header: 'School Email', key: 'schoolEmail'},
          { header: 'Primary Contact No', key: 'primaryContactNo'},
          { header: 'Secondary Contact No', key: 'secondaryContactNo'},
          { header: 'Additional Contact No', key: 'additionalContactNo'},
          { header: 'Established Year', key: 'established_year'},
          { header: 'Board', key: 'board'},
          { header: 'Address', key: 'address'},
          { header: 'City', key: 'city'},
          { header: 'State', key: 'state'},
          { header: 'Country', key: 'country'},
          { header: 'Pincode', key: 'pin'},
          { header: 'Max Class Limit', key: 'classLimit'},
          { header: 'Students Per Class', key: 'studentPerClass'},
          // Add other columns
        ];
  
        console.log(data[0])
  
        // const rowsToAdd = data.map(row => ({
        //     ...row, // Spread existing data properties
        //     assignedDate: new Date(row.assignedDate).toLocaleDateString('en-Gb'),
        //     deadline: new Date(row.deadline).toLocaleDateString('en-Gb'),
        //     assignedBy: row.assignedBy.name,
        //     assignedTo: (row.assignedTo).map((assignee) => assignee.name).join(', '),
        // }));
      
        // Add data (which is already processed for capitalization)
        // worksheet.addRows(data);
      
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
      
        // // Apply styling to data rows
        // worksheet.eachRow({ includeEmpty: false }, (row, rowNumber) => {
        //   if (rowNumber > 1) { // Skip header row
        //     row.eachCell((cell) => {
        //       cell.font = { name: 'Arial', size: 12 };
        //       cell.border = {
        //           top: {style:'thin'},
        //           left: {style:'thin'},
        //           bottom: {style:'thin'},
        //           right: {style:'thin'}
        //       };
        //       cell.alignment = { vertical: 'middle', horizontal: 'left' };
        //     });
        //   }
        // });
      
        // Generate the Excel file buffer and save it
        const buffer = await workbook.xlsx.writeBuffer();
        const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        saveAs(blob, `${fileName}.xlsx`);
      };
      exportToExcel(schools,"template_for_school_data_import");
    }


  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  // async def expiry_link(time: int):
  //   Credentials["expiry_time"] = time
  //   return {'success': True}

  const handleLinkConfig = async (value) => {
    if(value){
      const expiryLink = await fetch(`http://127.0.0.1:8000/users/expiry?time=${Number(value)}`,{
        method: 'POST',
      })
      if(expiryLink.ok){
        const response = await expiryLink.json();
        alert('Link Configured Successfully')
        window.location.reload()
      }
    }
  }

  const notificationOptions = [
    { label: 'before 7 days', value: 'before 7 days' },
    { label: 'before 2 days', value: 'before 2 days' },
    { label: 'after 1 days', value: 'after 1 days' },
    { label: 'after 2 days', value: 'after 2 days' },  
    { label: 'after 3 days', value: 'after 3 days' },
    { label: 'after 5 days', value: 'after 5 days' },
    { label: 'after 10 days', value: 'after 10 days' },
  ];

  const [selectedOption, setSelectedOption] = useState(null);

  const customStyles = {
    control: (provided) => ({
      ...provided,
      backgroundColor: 'white',
      borderColor: '#d1d5db', // Tailwind gray-300
      borderRadius: '0.375rem', // Tailwind rounded-md
      minHeight: '38px',
      boxShadow: 'none',
      '&:hover': {
        borderColor: '#9ca3af', // Tailwind gray-400
      },
    }),
    option: (provided, state) => ({
      ...provided,
      backgroundColor: state.isSelected ? '#eff6ff' : 'white', // Tailwind blue-50
      color: state.isSelected ? '#1e40af' : '#1f2937', // Tailwind blue-800 / gray-900
      '&:hover': {
        backgroundColor: '#f3f4f6', // Tailwind gray-100
      },
    }),
    singleValue: (provided) => ({
      ...provided,
      color: '#1f2937', // Tailwind gray-900
    }),
  };

  if (selectedSchool) {
    

    // Filter and paginate teachers
    let filteredTeachers = allTeachers.filter(t => 
      t.teacherName.toLowerCase().includes(teacherSearch.toLowerCase()) ||
      t.subject.toLowerCase().includes(teacherSearch.toLowerCase())
    );
    filteredTeachers = filteredTeachers.filter(t => t.schoolId === selectedSchool.schoolId);
    const teacherTotalPages = Math.ceil(filteredTeachers.length / 10);
    const paginatedTeachers = filteredTeachers.slice((teacherPage - 1) * 10, teacherPage * 10);

    // Filter and paginate students
    const filteredStudents = allStudents.filter(s => 
      s.schoolId === selectedSchool.schoolId &&
      (s.studentName.toLowerCase().includes(studentSearch.toLowerCase()) ||
      s.grade.toLowerCase().includes(studentSearch.toLowerCase()))
    );
    const studentTotalPages = Math.ceil(filteredStudents.length / 10);
    const paginatedStudents = filteredStudents.slice((studentPage - 1) * 10, studentPage * 10);

    console.log(students);

    

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 mb-6">
            <button
              onClick={() => setSelectedSchool(null)}
              className="text-blue-600 hover:text-blue-700 font-medium mb-4 flex items-center gap-2"
            >
              ← Back to Dashboard
            </button>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-slate-800">{selectedSchool.schoolName}</h1>
                {/* Location join by , city, state, pin */}
                <p className="text-slate-600 text-sm mt-1">{selectedSchool.city}, {selectedSchool.state} - {selectedSchool.pin} • Est. {selectedSchool.established_year}</p>
              </div>
              <button
                onClick={exportData}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <Download size={18} />
                Export Data
              </button>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-600 font-medium">Total Students</p>
                <p className="text-2xl font-bold text-blue-700">{allStudents.filter(s => s.schoolId === selectedSchool?.schoolId).length}</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-green-600 font-medium">Total Teachers</p>
                <p className="text-2xl font-bold text-green-700">{allTeachers.filter(t => t.schoolId === selectedSchool?.schoolId).length}</p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <p className="text-sm text-purple-600 font-medium">Avg Score</p>
                <p className="text-2xl font-bold text-purple-700">{selectedSchool.avgScore}%</p>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="bg-white rounded-xl shadow-sm mb-6">
            <div className="flex border-b overflow-x-auto">
              <button
                onClick={() => setActiveTab('teachers')}
                className={`px-6 py-4 font-medium whitespace-nowrap ${activeTab === 'teachers' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-600'}`}
              >
                Teachers ({allTeachers.filter(t => t.schoolId === selectedSchool?.schoolId).length})
              </button>
              <button
                onClick={() => setActiveTab('students')}
                className={`px-6 py-4 font-medium whitespace-nowrap ${activeTab === 'students' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-600'}`}
              >
                Students ({allStudents.filter(s => s.schoolId === selectedSchool?.schoolId).length})
              </button>
            </div>

            <div className="p-4 md:p-6">
              {activeTab === 'teachers' && (
                <div>
                  <div className="flex flex-col sm:flex-row gap-4 mb-4">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={18} />
                      <input
                        type="text"
                        placeholder="Search teachers..."
                        value={teacherSearch}
                        onChange={(e) => setTeacherSearch(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div className="text-sm text-slate-600 flex items-center">
                      Showing {paginatedTeachers.length} of {filteredTeachers.length}
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[600px]">
                      <thead className="bg-slate-50">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">On-board Date</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Name</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Qualification</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Role</th>
                          {/* <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Subjects</th> */}
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Status</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        {paginatedTeachers.map(teacher => (
                          <tr key={teacher.id} className="hover:bg-slate-50">
                            <td className="px-4 py-3 text-sm text-slate-800">{new Date(teacher.onboardingDate).toLocaleDateString('en-Gb')}</td>
                            <td className="px-4 py-3 text-sm text-slate-800">{teacher.teacherName}</td>
                            <td className="px-4 py-3 text-sm text-slate-600">{teacher.qualification}</td>
                            <td className="px-4 py-3 text-sm text-slate-600">{roles.map(role => role.roleId === teacher.role ? role.roleName : null)}</td>
                            {/* <td className="px-4 py-3 text-sm text-slate-600">{subjects.map(subject => subject.subjectId === teacher.subjectId ? subject.subjectName : '-')}</td> */}
                            <td className="px-4 py-3">
                              <span className={`px-2 py-1 text-xs rounded-full ${teacher.active === true ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                                {teacher.active === true ? 'Active' : 'Inactive'}
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <button
                                onClick={() => handleDiscontinue('Teacher', teacher)}
                                className="text-red-600 hover:text-red-700 text-sm font-medium"
                              >
                                Discontinue
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Teacher Pagination */}
                  {teacherTotalPages > 1 && (
                    <div className="flex items-center justify-between mt-4 flex-wrap gap-4">
                      <p className="text-sm text-slate-600">
                        Page {teacherPage} of {teacherTotalPages}
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setTeacherPage(Math.max(1, teacherPage - 1))}
                          disabled={teacherPage === 1}
                          className="px-3 py-1 border border-slate-300 rounded-lg disabled:opacity-50 hover:bg-slate-50"
                        >
                          <ChevronLeft size={18} />
                        </button>
                        <button
                          onClick={() => setTeacherPage(Math.min(teacherTotalPages, teacherPage + 1))}
                          disabled={teacherPage === teacherTotalPages}
                          className="px-3 py-1 border border-slate-300 rounded-lg disabled:opacity-50 hover:bg-slate-50"
                        >
                          <ChevronRight size={18} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'students' && (
                <div>
                  <div className="flex flex-col sm:flex-row gap-4 mb-4">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={18} />
                      <input
                        type="text"
                        placeholder="Search students..."
                        value={studentSearch}
                        onChange={(e) => setStudentSearch(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div className="text-sm text-slate-600 flex items-center">
                      Showing {paginatedStudents.length} of {filteredStudents.length}
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[600px]">
                      <thead className="bg-slate-50">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Name</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Date of Birth</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Gender</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Class</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Status</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        {paginatedStudents.map(student => (
                          <tr key={student.id} className="hover:bg-slate-50">
                            <td className="px-4 py-3 text-sm text-slate-800">{student.studentName}</td>
                            <td className="px-4 py-3 text-sm text-slate-600">{new Date(student.DOB).toLocaleDateString('en-Gb')}</td>
                            <td className="px-4 py-3 text-sm text-slate-600">{student.gender}</td>
                            <td className="px-4 py-3 text-sm text-slate-600">{classes.map(cls => cls.classId === student.classId ? cls.className : null)}</td>
                            <td className="px-4 py-3">
                              <span className={`px-2 py-1 text-xs rounded-full ${student.active === true ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                                {student.active === true ? 'Active' : 'Inactive'}
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <button
                                onClick={() => handleDiscontinue('Student', student)}
                                className="text-red-600 hover:text-red-700 text-sm font-medium"
                              >
                                Discontinue
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Student Pagination */}
                  {studentTotalPages > 1 && (
                    <div className="flex items-center justify-between mt-4 flex-wrap gap-4">
                      <p className="text-sm text-slate-600">
                        Page {studentPage} of {studentTotalPages}
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setStudentPage(Math.max(1, studentPage - 1))}
                          disabled={studentPage === 1}
                          className="px-3 py-1 border border-slate-300 rounded-lg disabled:opacity-50 hover:bg-slate-50"
                        >
                          <ChevronLeft size={18} />
                        </button>
                        <button
                          onClick={() => setStudentPage(Math.min(studentTotalPages, studentPage + 1))}
                          disabled={studentPage === studentTotalPages}
                          className="px-3 py-1 border border-slate-300 rounded-lg disabled:opacity-50 hover:bg-slate-50"
                        >
                          <ChevronRight size={18} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Discontinue Modal */}
        {discontinueModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl p-6 max-w-md w-full">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-slate-800">Confirm Discontinue</h3>
                <button onClick={() => setDiscontinueModal(null)} className="text-slate-400 hover:text-slate-600">
                  <X size={24} />
                </button>
              </div>
              <p className="text-slate-600 mb-6">
                Are you sure you want to discontinue <strong>{discontinueModal.type}</strong>{' '}
                <strong>"{discontinueModal.item.name}"</strong>?
              </p>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDiscontinueModal(null)}
                  className="px-4 py-2 border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDiscontinue}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Discontinue
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  

  return (
    (loading) ? (<Loader />) : (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-6">
      {popUpMsg && (
        <div className="fixed top-4 right-4 z-50 animate-fade-in">
          <div className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg">
            {popUpMsg}
          </div>
        </div>
      )}
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className='flex items-center justify-between'>
          <div className="mb-6 md:mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-2">Admin Dashboard</h1>
            <p className="text-slate-600">Manage {schools.length} schools, teachers, students, and view performance analytics</p>
          </div>
          <div className='flex gap-4'>
            <button
              onClick={() => setImportModal(true)}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Upload size={20} />
              Import School Data
            </button>
            <button
              onClick={() => navigate('/add-school')}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium"
            >
              <Plus size={20} />
              Add School
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

        {/* Stats Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-6 md:mb-8">
          <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs md:text-sm text-slate-600 font-medium">Total Schools</p>
                <p className="text-2xl md:text-3xl font-bold text-slate-800 mt-1">{schools.length}</p>
              </div>
              <div className="bg-blue-100 p-2 md:p-3 rounded-lg">
                <School className="text-blue-600" size={20} />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs md:text-sm text-slate-600 font-medium">Total Students</p>
                <p className="text-2xl md:text-3xl font-bold text-slate-800 mt-1">
                  {allStudents.length}
                </p>
              </div>
              <div className="bg-green-100 p-2 md:p-3 rounded-lg">
                <GraduationCap className="text-green-600" size={20} />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs md:text-sm text-slate-600 font-medium">Total Teachers</p>
                <p className="text-2xl md:text-3xl font-bold text-slate-800 mt-1">
                  {allTeachers.length}
                </p>
              </div>
              <div className="bg-purple-100 p-2 md:p-3 rounded-lg">
                <Users className="text-purple-600" size={20} />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs md:text-sm text-slate-600 font-medium">Avg Performance</p>
                <p className="text-2xl md:text-3xl font-bold text-slate-800 mt-1">
                  {schools.length > 0 ? (schools.reduce((total, school) => total + school.avgScore, 0) / schools.length).toFixed(2) : 0}%
                </p>
              </div>
              <div className="bg-orange-100 p-2 md:p-3 rounded-lg">
                <TrendingUp className="text-orange-600" size={20} />
              </div>
            </div>
          </div>
        </div>

        {/* Schools Section */}
        <div className="bg-white rounded-xl shadow-sm p-4 md:p-6 mb-6 md:mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
            <h2 className="text-xl md:text-2xl font-bold text-slate-800">Schools Directory</h2>
            
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1 sm:flex-initial">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={18} />
                <input
                  type="text"
                  placeholder="Search schools..."
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="w-full sm:w-64 pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50"
              >
                <Filter size={18} />
                Filters
              </button>
              
              <button
                onClick={exportData}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <Download size={18} />
                Export Data
              </button>
              <button onClick={() => navigate('/audit')} className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Audit</button>
            </div>
          </div>

          {/* Filter Panel */}
          {showFilters && (
            <div className="mb-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
              <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                {/* <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">Sort By</label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="name">Name</option>
                    <option value="students">Students Count</option>
                    <option value="teachers">Teachers Count</option>
                    <option value="avgScore">Average Score</option>
                    <option value="established">Year Established</option>
                  </select>
                </div> */}
                
                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">Order</label>
                  <select
                    value={sortOrder}
                    onChange={(e) => setSortOrder(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="asc">Ascending</option>
                    <option value="desc">Descending</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">Status</label>
                  <select
                    value={filterStatus}
                    onChange={(e) => {
                      setFilterStatus(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">Performance</label>
                  <select
                    value={filterPerformance}
                    onChange={(e) => {
                      setFilterPerformance(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Levels</option>
                    <option value="high">High (80%+)</option>
                    <option value="medium">Medium (70-79%)</option>
                    <option value="low">Low (&lt;70%)</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">City</label>
                  <select
                    value={filterCity}
                    onChange={(e) => {
                      setFilterCity(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Cities</option>
                    {schools.map((school) => (
                      <option key={school.city} value={school.city}>{school.city}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">State</label>
                  <select
                    value={filterState}
                    onChange={(e) => {
                      setFilterState(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All States</option>
                    {schools.map((school) => (
                      <option key={school.state} value={school.state}>{school.state}</option>
                    ))}
                  </select>
                </div>    

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">Pincode</label>
                  <select
                    value={filterPincode}
                    onChange={(e) => {
                      setFilterPincode(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Pincodes</option>
                    {schools.map((school) => (
                      <option key={school.pin} value={school.pin}>{school.pin}</option>
                    ))}
                  </select>
                </div>    
              </div>
              
            </div>
          )}

          {/* Results Summary */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4 text-left">
            <div className='flex items-center gap-4'>
              <div className="">
                <label htmlFor="notification-select" className="block text-sm font-medium text-gray-700 mb-2">
                  Select Fee Notification Preferences:
                </label>
                <Select
                  isMulti
                  id="notification-select"
                  options={notificationOptions}
                  onChange={setSelectedOption}
                  value={selectedOption}
                  styles={customStyles}
                  className="w-full"
                  placeholder="Choose a notification time..."
                />
              </div>

              <div>
                <label htmlFor="notification-select" className="block text-sm font-medium text-gray-700 mb-2">
                  Set Reset Password Link Expiration:
                </label>
                <select
                  onChange={(e) => {
                    handleLinkConfig(e.target.value);
                  }}
                  className="w-64 px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                >
                  <option value="">select time</option>
                  <option value="3">3 minutes</option>
                  <option value="5">5 minutes</option>
                  <option value="10">10 minutes</option>
                  <option value="15">15 minutes</option>
                  <option value="20">20 minutes</option>
                  <option value="30">30 minutes</option>
                  <option value="45">45 minutes</option>
                  <option value="60">1 hour</option>
                </select>
              </div>
            </div>
          </div>

          <div className='flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4'>
            <p className="text-sm text-slate-600">
              Showing {paginatedSchools.length} of {filteredAndSortedSchools.length} schools
            </p>
            <div className=''>
              <label htmlFor="notification-select" className="block text-sm font-medium text-gray-700 mb-2">
                Show results per page:
              </label>
              <select
                value={itemsPerPage}
                onChange={(e) => {
                  setItemsPerPage(Number(e.target.value));
                  setCurrentPage(1);
                }}
                className="w-32 px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="8">8 per page</option>
                <option value="12">12 per page</option>
                <option value="24">24 per page</option>
                <option value="48">48 per page</option>
              </select>
            </div>
          </div>
          

          {/* Schools Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
            {paginatedSchools.map(school => (
              <div
                key={school.id}
                onClick={() => handleSchoolClick(school)}
                className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 cursor-pointer hover:shadow-lg transition-all transform hover:-translate-y-1 border border-blue-200"
              >
                <div className="flex items-center justify-between mb-3">
                  <School className="text-blue-600" size={24} />
                  <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                    school.status === 'active' 
                      ? 'bg-green-100 text-green-700' 
                      : 'bg-gray-100 text-gray-700'
                  }`}>
                    {school.status}
                  </span>
                </div>
                <h3 className="text-base font-bold text-slate-800 mb-2 line-clamp-2 min-h-[3rem]">
                  {school.schoolName}
                </h3>
                <p className="text-xs text-slate-600 mb-3">{school.address},{school.city}, {school.state}-{school.pin}</p>
                <div className="space-y-2">
                <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Board:</span>
                    <span className="font-semibold text-slate-800">{school.board}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Students:</span>
                    <span className="font-semibold text-slate-800">{allStudents.map(student => student.schoolId === school?.schoolId).reduce((a, b) => a + b, 0)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Teachers:</span>
                    <span className="font-semibold text-slate-800">{allTeachers.map(teacher => teacher.schoolId === school?.schoolId).reduce((a, b) => a + b, 0)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Avg Score:</span>
                    <span className={`font-semibold ${
                      school.avgScore >= 80 ? 'text-green-600' :
                      school.avgScore >= 70 ? 'text-blue-600' : 'text-orange-600'
                    }`}>
                      {school.avgScore?.toFixed(1) || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Max Class Limit:</span>
                    <span className="font-semibold text-slate-800">{school.maxClassLimit}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Student Per Class:</span>
                    <span className="font-semibold text-slate-800">{school.studentsPerClass}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>


          {filteredAndSortedSchools.length === 0 && (
            <div className="text-center py-12">
              <AlertCircle className="mx-auto text-slate-400 mb-4" size={48} />
              <p className="text-slate-600">No schools found matching your criteria</p>
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t">
              <p className="text-sm text-slate-600">
                Page {currentPage} of {totalPages}
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentPage(1)}
                  disabled={currentPage === 1}
                  className="px-3 py-2 border border-slate-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 text-sm"
                >
                  First
                </button>
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-2 border border-slate-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                >
                  <ChevronLeft size={18} />
                </button>
                
                {/* Page Numbers */}
                <div className="hidden sm:flex gap-2">
                  {[...Array(Math.min(5, totalPages))].map((_, i) => {
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
                        className={`px-3 py-2 rounded-lg text-sm ${
                          currentPage === pageNum
                            ? 'bg-blue-600 text-white'
                            : 'border border-slate-300 hover:bg-slate-50'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-2 border border-slate-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                >
                  <ChevronRight size={18} />
                </button>
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="px-3 py-2 border border-slate-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 text-sm"
                >
                  Last
                </button>
              </div>
            </div>
          )}
        </div>

        {importModal && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                        <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
                          <div className="bg-green-600 text-white p-4 flex items-center justify-between">
                            <h2 className="text-xl font-bold">Import Students</h2>
                            <button
                              onClick={() => {
                                setImportModal(false);
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
                                  onClick={downloadTemplate}
                                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                                >
                                  <Download size={18} />
                                  Download School Import Template
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
          <EducationDashboard />
      </div>
    </div>
    )
  );
};

export default AdminDashboard2;