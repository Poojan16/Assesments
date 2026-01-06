# Leave Management System - Frontend

A modern, professional frontend for the Leave Management System built with React, Vite, and Tailwind CSS.

## Features

- 🎨 Professional and modern UI/UX design
- 🔐 User authentication (Login & Sign Up)
- ✅ Form validation
- 📱 Responsive design
- 🎯 Smooth animations and transitions

## Getting Started

### Prerequisites

- Node.js (v20.18.3 or higher recommended)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

### Running the Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Login.jsx      # Login page component
│   │   └── SignUp.jsx      # Sign up page component
│   ├── App.jsx             # Main app component with routing
│   ├── main.jsx            # Entry point
│   └── index.css           # Global styles with Tailwind
├── public/                 # Static assets
└── package.json            # Dependencies and scripts
```

## Pages

### Login Page (`/login`)
- Email and password authentication
- Remember me functionality
- Forgot password link
- Form validation

### Sign Up Page (`/signup`)
- Employee registration form
- Fields: First Name, Last Name, Email, Employee ID, Department, Password
- Password confirmation
- Terms and conditions acceptance
- Comprehensive form validation

## Technologies Used

- **React 19** - UI library
- **Vite** - Build tool and dev server
- **React Router DOM** - Client-side routing
- **Tailwind CSS v4** - Utility-first CSS framework

## Next Steps

- Connect to backend API
- Implement authentication flow
- Add dashboard page
- Add leave request functionality
