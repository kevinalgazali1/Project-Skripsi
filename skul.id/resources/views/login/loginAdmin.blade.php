<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />

    <style>
        html,
        body {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: "Poppins", sans-serif;
            background-color: #fff;
            color: #252f35;
            font-size: 16px;
        }

        .full-page-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            padding: 40px;
            position: relative;
        }

        .btn-back {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 999;
        }

        .logo-wrapper {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 40px;
        }

        .logo-wrapper img {
            height: 50px;
        }

        .inner-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 100px;
            flex-wrap: wrap;
            width: 100%;
            max-width: 1000px;
        }

        .icon-besar {
            flex: 1;
            min-width: 300px;
            text-align: center;
        }

        .icon-besar img {
            max-width: 100%;
            height: auto;
        }

        .login-form {
            flex: 1;
            min-width: 300px;
        }

        h1 {
            font-weight: 600;
            font-size: 36px;
            margin-bottom: 15px;
        }

        p {
            font-size: 18px;
        }

        .form-control {
            background-color: #f1f8f7;
            border: none;
            padding-left: 55px;
            height: 70px;
            font-size: 18px;
        }

        .form-control:focus {
            box-shadow: none;
        }

        .input-group-text {
            background: none;
            border: none;
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #999;
            font-size: 22px;
            z-index: 2;
        }

        .form-check-input {
            width: 20px;
            height: 20px;
            margin-top: 5px;
        }

        .form-check-label {
            font-size: 16px;
        }

        .form-check a {
            text-decoration: underline;
            font-weight: 500;
        }

        .form-check-input:checked {
            background-color: #ff3c6a;
            border-color: #ff3c6a;
        }

        .login-btn {
            background: #00ffe1;
            color: white;
            font-weight: 600;
            font-size: 20px;
            height: 70px;
        }

        .login-btn:hover {
            background-color: #00FFE1 !important;
            color: white !important;
            box-shadow: none !important;
            opacity: 1 !important;
            transform: none !important;
        }

        .text-danger {
            font-size: 14px;
        }

        .alert {
            font-size: 16px;
        }

        @media (max-width: 768px) {
            .inner-content {
                flex-direction: column;
                text-align: center;
            }

            .icon-besar img {
                width: 300px;
            }

            h1 {
                font-size: 30px;
            }

            p,
            .form-check-label {
                font-size: 16px;
            }

            .form-control,
            .login-btn {
                font-size: 16px;
                height: 60px;
            }

            .full-page-container {
                padding: 20px;
            }

            .logo-wrapper {
                justify-content: center;
                padding: 0 20px;
                margin-bottom: 30px;
            }

            .btn-back {
                top: 15px;
                left: 15px;
            }

            button {
                margin-bottom: 80px;
            }

            .log-logo {
                margin-top: 150px;
            }
        }
    </style>
</head>

<body>
    <div class="full-page-container">
        <div class="inner-content">
            <div class="icon-besar">
                <img src="{{ url('img/login-mitra.png') }}" alt="Icon Mitra" class="log-logo" />
            </div>

            <div class="login-form">
                <div class="text mb-3">
                    <h1>Login Admin</h1>
                </div>

                <form action="{{ route('login') }}" method="POST">
                    @csrf
                    <input type="hidden" name="login_type" value="admin" />

                    <!-- Nomor HP -->
                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-phone"></i></span>
                        <input type="text" class="form-control" placeholder="Nomor Handphone" name="identifier"
                            required />
                        <div class="text-danger mt-1 small" id="phoneError"></div>
                    </div>

                    <!-- Password -->
                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-lock"></i></span>
                        <input type="password" placeholder="Password" class="form-control" name="password"
                            id="passwordInput" minlength="8" required />
                        <span class="position-absolute top-50 end-0 translate-middle-y pe-3" style="cursor:pointer;"
                            onclick="togglePassword('passwordInput', 'toggleIcon')">
                            <i class="bi bi-eye-slash" id="toggleIcon"></i>
                        </span>
                        <div class="text-danger mt-1 small" id="passwordError"></div>
                    </div>

                    <div class="mb-3 text-end">
                        <a href="#" class="text-danger small-link">Lupa Password?</a>
                    </div>

                    <div class="form-check mb-3 text-start">
                        <input class="form-check-input" type="checkbox" id="termsCheck" required />
                        <label class="form-check-label" for="termsCheck">
                            Dengan login, saya setuju dengan
                            <a href="#">Ketentuan Pengguna</a> & <a href="#">Kebijakan Privasi</a>
                        </label>
                    </div>

                    <button type="submit" class="btn login-btn w-100">LOGIN</button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
        function togglePassword(id, iconId) {
            const input = document.getElementById(id);
            const icon = document.getElementById(iconId);
            if (input.type === "password") {
                input.type = "text";
                icon.classList.remove("bi-eye-slash");
                icon.classList.add("bi-eye");
            } else {
                input.type = "password";
                icon.classList.remove("bi-eye");
                icon.classList.add("bi-eye-slash");
            }
        }

        document.querySelector("form").addEventListener("submit", function(e) {
            const phoneInput = document.querySelector('input[name="identifier"]');
            const phoneError = document.getElementById("phoneError");
            const passwordInput = document.querySelector('input[name="password"]');
            const passwordError = document.getElementById("passwordError");

            phoneError.textContent = "";
            passwordError.textContent = "";

            const phoneValue = phoneInput.value.trim();
            const passwordValue = passwordInput.value.trim();

            let hasError = false;

            if (!/^08\d{8,13}$/.test(phoneValue)) {
                phoneError.textContent = "Nomor handphone tidak valid.";
                hasError = true;
            }

            if (passwordValue.length < 8) {
                passwordError.textContent = "Password minimal 8 karakter.";
                hasError = true;
            }

            if (hasError) e.preventDefault();
        });

        window.addEventListener("DOMContentLoaded", function() {
            const alerts = document.querySelectorAll(".alert");
            alerts.forEach(alert => {
                setTimeout(() => {
                    alert.style.transition = "opacity 0.5s ease";
                    alert.style.opacity = "0";
                    setTimeout(() => alert.remove(), 500);
                }, 3000);
            });
        });
    </script>
    <script>
        @if (session('success'))
            Swal.fire({
                icon: 'success',
                title: 'Berhasil!',
                text: '{{ session('success') }}',
                timer: 3000,
                showConfirmButton: false
            });
        @endif

        @if (session('login_error'))
            Swal.fire({
                icon: 'error',
                title: 'Login Gagal',
                text: {!! json_encode(session('login_error')) !!},
                timer: 3000,
                showConfirmButton: false
            });
        @endif

        @if (session('message'))
            Swal.fire({
                icon: '{{ session('alert-type') == 'warning' ? 'warning' : 'info' }}',
                title: '{{ ucfirst(session('alert-type') ?? 'Info') }}',
                text: '{{ session('message') }}',
                timer: 3000,
                showConfirmButton: false
            });
        @endif
    </script>
</body>

</html>
