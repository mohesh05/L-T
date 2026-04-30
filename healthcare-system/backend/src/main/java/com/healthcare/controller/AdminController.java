package com.healthcare.controller;
import com.healthcare.dto.AuthDtos.DoctorReq;import com.healthcare.entity.Doctor;import com.healthcare.service.impl.AppService;import jakarta.validation.Valid;import lombok.RequiredArgsConstructor;import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/admin") @RequiredArgsConstructor public class AdminController { private final AppService s; @PostMapping("/doctors") public Doctor add(@Valid @RequestBody DoctorReq r){ return s.addDoctor(r);} }
