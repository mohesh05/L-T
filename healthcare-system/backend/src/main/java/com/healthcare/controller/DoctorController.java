package com.healthcare.controller;
import com.healthcare.dto.AuthDtos.StatusReq;import com.healthcare.entity.Appointment;import com.healthcare.service.impl.AppService;import jakarta.validation.Valid;import lombok.RequiredArgsConstructor;import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/doctor") @RequiredArgsConstructor public class DoctorController { private final AppService s;
@PatchMapping("/appointments/{id}/status") public Appointment status(@PathVariable Long id,@Valid @RequestBody StatusReq r){ return s.updateStatus(id,r);} }
