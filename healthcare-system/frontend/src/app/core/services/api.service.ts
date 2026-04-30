import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
@Injectable({providedIn:'root'})
export class ApiService{base='http://localhost:8080'; constructor(private http:HttpClient){}
headers(){const t=localStorage.getItem('token'); return {headers:new HttpHeaders({'Authorization':`Bearer ${t}`})};}
login(body:any){return this.http.post(`${this.base}/api/auth/login`,body)}
register(body:any){return this.http.post(`${this.base}/api/auth/register`,body)}
doctors(){return this.http.get(`${this.base}/api/patient/doctors`,this.headers())}
book(body:any){return this.http.post(`${this.base}/api/patient/appointments`,body,this.headers())}
}
