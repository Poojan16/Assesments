# Leave Management System - Frontend

A modern, professional frontend for the Leave Management System built with React 19, Vite, and Tailwind CSS. Features a responsive design with smooth animations and a complete authentication flow.

## Features

- 🎨 Modern and professional UI/UX design
- 🔐 Complete authentication flow (Login & Sign Up)
- 📊 Interactive Dashboard with leave statistics
- 📝 Leave application form with validation
- 📋 My Leaves page with status tracking
- 👥 Manager dashboard for team leave management
- ✅ Role-based access control (Employee vs Manager)
- 📱 Fully responsive design
- 🎯 Smooth animations and transitions
- 🔄 Real-time API integration

## Tech Stack

- **React 19** - Modern UI library
- **Vite** - Lightning-fast build tool and dev server
- **React Router DOM v7** - Client-side routing
- **Tailwind CSS v4** - Utility-first CSS framework
- **Axios** - HTTP client for API requests
- **Heroicons & Lucide React** - Beautiful icon library
- **ESLint** - Code linting

## Getting Started

### Prerequisites

- Node.js (v20.18.3 or higher recommended)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

### Running the Development Server

```bash
npm run dev
```

The application will be available at **http://localhost:5173**

### Building for Production

```bash
npm run build
```

The production-ready files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

---

## Project Structure

```
frontend/
├── public/                 # Static assets (favicon, images, etc.)
├── src/
│   ├── assets/            # Static assets (images, icons)
│   ├── components/        # Reusable components
│   │   └── ProtectedRoute.jsx    # Route protection component
│   ├── contexts/          # React contexts
│   │   └── AuthContext.jsx       # Authentication context & provider
│   ├── layouts/           # Page layouts
│   │   └── DashboardLayout.jsx   # Main dashboard layout
│   ├── pages/             # Page components
│   │   ├── Login.jsx              # User login page
│   │   ├── SignUp.jsx             # User registration page
│   │   ├── Dashboard.jsx          # Main dashboard with statistics
│   │   ├── ApplyLeave.jsx         # Leave application form
│   │   ├── MyLeaves.jsx           # Employee's leave history
│   │   └── ManagerRequests.jsx    # Manager's team leave requests
│   ├── services/          # API services
│   │   └── api.js                 # Axios API client configuration
│   ├── App.css            # Global styles
│   ├── App.jsx            # Main app component with routing
│   ├── main.jsx           # Application entry point
│   └── index.css          # Global styles with Tailwind
├── index.html             # HTML template
├── package.json           # Dependencies and scripts
├── vite.config.js         # Vite configuration
├── tailwind.config.js     # Tailwind CSS configuration
├── postcss.config.js      # PostCSS configuration
├── eslint.config.js       # ESLint configuration
└── README.md              # This file
```

---

## Pages & Features

### 1. Login Page (`/login`)

**Features:**
- Email and password authentication
- Form validation
- Error handling with user-friendly messages
- Link to signup page

**Access:** Public (all users)

---

### 2. Sign Up Page (`/signup`)

**Features:**
- Employee registration form
- Fields: First Name, Last Name, Email, Employee ID, Department, Password
- Password confirmation
- Form validation
- Error handling for duplicates

**Access:** Public (all users)

**Departments Available:**
- IT
- Engineering
- HR
- Marketing
- Finance
- Sales
- Operations

---

### 3. Dashboard (`/`)

**Features:**
- Welcome message with user name
- Leave statistics overview
- Quick action buttons
- Recent leave applications
- Department-wise leave distribution
- Leave type distribution (Casual vs Sick)

**Access:** Authenticated users (Employee & Manager)

**Dashboard Stats:**
- Total Leaves
- Pending Requests
- Approved Leaves
- Rejected Leaves

---

### 4. Apply Leave (`/apply-leave`)

**Features:**
- Leave type selection (Casual/Sick)
- Start date and end date pickers
- Reason textarea
- Form validation
- Date range validation
- Submit and cancel buttons
- Success/error notifications

**Access:** Authenticated users (Employee only)

**Leave Types:**
- **Casual Leave** - For vacations, personal work, family events
- **Sick Leave** - For medical issues, doctor appointments

---

### 5. My Leaves (`/my-leaves`)

**Features:**
- List of all leave applications
- Status badges (Pending, Approved, Rejected)
- Leave type badges (Casual, Sick)
- Date range display
- Reason for leave
- Leave duration calculation
- Filtering by status
- Pagination support
- Color-coded status indicators

**Access:** Authenticated users (Employee only)

**Status Indicators:**
- 🟡 Pending - Yellow
- 🟢 Approved - Green
- 🔴 Rejected - Red

---

### 6. Manager Requests (`/manager-requests`)

**Features:**
- View all team leave requests
- Employee information (name, ID, department)
- Leave details (type, dates, reason)
- Approve/Reject actions
- Add remarks for decisions
- Filter by status
- Statistics overview
- Team performance metrics

**Access:** Authenticated users (Manager only)

**Manager Actions:**
- ✅ Approve leave with optional remarks
- ❌ Reject leave with required reason
- 📊 View team statistics

---

## Authentication Flow

### Login Process
1. User enters email and password
2. Frontend sends credentials to backend API
3. Backend validates and returns JWT token
4. Frontend stores token in localStorage
5. User is redirected to Dashboard

### Protected Routes

The `ProtectedRoute` component guards private routes:

```jsx
// Route protection example
<Route 
  path="/dashboard" 
  element={
    <ProtectedRoute>
      <Dashboard />
    </ProtectedRoute>
  } 
/>
```

### Auth Context

The `AuthContext` provides:

- **Authentication State:**
  - `user` - Current user information
  - `isAuthenticated` - Authentication status
  - `isLoading` - Loading state

- **Authentication Methods:**
  - `login(credentials)` - User login
  - `logout()` - User logout
  - `register(userData)` - User registration
  - `updateUser(userData)` - Update user profile

- **Token Management:**
  - Automatic token storage in localStorage
  - Token inclusion in API requests
  - Logout clears stored tokens

---

## API Integration

### API Client Configuration

The API client is configured in `src/services/api.js`:

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

### API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/signup` | POST | Register new user |
| `/auth/login` | POST | User login |
| `/leaves/apply` | POST | Apply for leave |
| `/leaves/my-leaves` | GET | Get user's leaves |
| `/leaves/{id}` | GET | Get leave by ID |
| `/manager/leaves` | GET | Get all team leaves (Manager) |
| `/manager/leaves/{id}/approve` | PUT | Approve leave (Manager) |
| `/manager/leaves/{id}/reject` | PUT | Reject leave (Manager) |
| `/manager/stats` | GET | Get team statistics (Manager) |

---

## Styling & Design

### Tailwind CSS

The project uses Tailwind CSS v4 for styling:

- **Utility-first approach** - Build designs directly in markup
- **Responsive design** - Mobile-first breakpoints
- **Dark mode ready** - Supports dark mode theming
- **Custom colors** - Brand colors configured

### Color Scheme

- **Primary:** Blue/Indigo
- **Success:** Green
- **Warning:** Yellow/Orange
- **Danger:** Red
- **Neutral:** Gray/Slate

### Components Styling

- **Buttons:** Primary, secondary, danger variants
- **Forms:** Input fields with validation states
- **Cards:** Shadow and border styling
- **Tables:** Responsive data tables
- **Badges:** Status and type indicators
- **Icons:** Heroicons integrated throughout

---

## Development

### Code Style

The project uses ESLint for code quality:

```bash
# Run linting
npm run lint
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

---

## Production Deployment

### Build Optimization

1. **Production Build:**
   ```bash
   npm run build
   ```

2. **Optimized Output:**
   - Minified JavaScript and CSS
   - Code splitting for faster loading
   - Asset optimization

### Deployment Checklist

- [ ] Set correct API base URL in production
- [ ] Configure CORS on backend for frontend domain
- [ ] Set up proper environment variables
- [ ] Enable HTTPS in production
- [ ] Configure proper caching headers
- [ ] Test all user flows
- [ ] Verify responsive design

### Example Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /path/to/frontend/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linting and tests
5. Submit a pull request

---

## License

MIT License

