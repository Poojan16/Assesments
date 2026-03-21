import { ROLES } from "./roles";

export const PERMISSIONS = {
    TASK_CREATE: 'task.create',
    TASK_EDIT: 'task.edit',
    TASK_VIEW: 'task.view',
    PM_HOME: 'PM.home',
    TE_HOME: 'TE.home'
  };

  export const ROLE_PERMISSIONS = {
    [ROLES.PM]: [
      PERMISSIONS.TASK_CREATE,
      PERMISSIONS.TASK_EDIT,
      PERMISSIONS.TASK_VIEW,
      PERMISSIONS.PM_HOME,
    ],
    [ROLES.TE]: [
      PERMISSIONS.TE_HOME,
      PERMISSIONS.TASK_EDIT,
      PERMISSIONS.TASK_VIEW,
    ],
  };
