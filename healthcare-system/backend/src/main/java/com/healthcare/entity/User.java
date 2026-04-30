package com.healthcare.entity;
import com.healthcare.enums.Role;import jakarta.persistence.*;import lombok.*;
@Entity @Table(name="users",uniqueConstraints=@UniqueConstraint(columnNames="email")) @Getter @Setter @NoArgsConstructor @AllArgsConstructor
public class User { @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id; private String username; private String email; private String password; @Enumerated(EnumType.STRING) private Role role; }
