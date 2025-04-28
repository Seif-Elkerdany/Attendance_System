document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.classList.add('fade');
            setTimeout(() => msg.remove(), 300);
        }, 5000);
    });


    const courseSelect = document.getElementById('course-select');
    if (courseSelect) {
        courseSelect.addEventListener('change', function() {
            const selectedCourse = this.value;
            console.log('Selected course:', selectedCourse);
        });
    }

    
    const startCameraBtn = document.getElementById('start-camera');
    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', function() {
            alert('Camera functionality will be integrated with the face recognition system');
        });
    }

// Password strength indicator
document.getElementById('password')?.addEventListener('input', function() {
    const password = this.value;
    const strengthIndicator = document.getElementById('password-strength');
    const requirements = document.querySelectorAll('.password-requirements li');
    
    if (!strengthIndicator) return;
    
    // Reset classes
    strengthIndicator.className = 'password-strength';
    requirements.forEach(req => req.classList.remove('requirement-met'));
    
    // Check requirements
    const hasMinLength = password.length >= 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    // Update requirements list
    if (hasMinLength) requirements[0].classList.add('requirement-met');
    if (hasUpperCase) requirements[1].classList.add('requirement-met');
    if (hasLowerCase) requirements[2].classList.add('requirement-met');
    if (hasNumbers) requirements[3].classList.add('requirement-met');
    if (hasSpecial) requirements[4].classList.add('requirement-met');
    
    // Calculate strength
    let strength = 0;
    if (password.length > 0) strength += 1;
    if (password.length >= 8) strength += 1;
    if (hasUpperCase) strength += 1;
    if (hasNumbers) strength += 1;
    if (hasSpecial) strength += 1;
    
    // Update strength indicator
    strengthIndicator.classList.add(`strength-${strength}`);
});
});