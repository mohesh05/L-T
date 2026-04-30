import { CanActivateFn } from '@angular/router';
export const roleGuard=(role:string):CanActivateFn=>()=> (localStorage.getItem('role')===role);
