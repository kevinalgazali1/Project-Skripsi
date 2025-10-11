<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Register - Skul.Id</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />

    <style>
        body {
            font-family: "Poppins", sans-serif;
            color: #252f35;
            font-size: 13px;
            background-color: #fff;
            margin: 0;
            padding: 0;
        }

        .btn-back {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 999;
        }

        .container-box {
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            max-width: 900px;
            margin: 100px auto;
            padding: 30px;
            position: relative;
        }

        .logo-wrapper img {
            height: 40px;
            object-fit: contain;
        }

        .inner-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 30px;
            margin-top: 50px;
            flex-wrap: wrap;
        }

        .icon-besar {
            flex: 1;
            min-width: 250px;
            text-align: center;
        }

        .icon-besar img {
            max-width: 100%;
            height: auto;
        }

        .login-form {
            flex: 1;
            min-width: 250px;
        }

        input {
            height: 60px;
        }

        h1 {
            font-weight: 600;
            font-size: 24px;
            margin-bottom: 10px;
        }

        p {
            font-size: 14px;
        }

        .form-control {
            background-color: #f1f8f7;
            border: none;
            padding-left: 40px;
        }

        .form-control:focus {
            box-shadow: none;
        }

        .input-group-text {
            background: none;
            border: none;
            position: absolute;
            left: 5px;
            top: 50%;
            transform: translateY(-50%);
            color: #999;
        }

        .form-check-label {
            font-size: 13px;
        }

        .form-check a {
            text-decoration: underline;
        }

        .form-check-input:checked {
            background-color: #ff3c6a;
            border-color: #ff3c6a;
        }

        .login-btn {
            background: #00FFE1;
            color: white;
            font-weight: 600;
            height: 60px;
        }

        .login-btn:hover {
            background-color: #00FFE1 !important;
            color: white !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        @media (max-width: 768px) {
            .inner-content {
                flex-direction: column;
                text-align: center;
            }

            .logo-wrapper {
                justify-content: center !important;
                margin-bottom: 20px;
            }
        }
    </style>
</head>

<body>

    <!-- Tombol Back -->
    <a href="{{ url()->previous() }}" class="btn-back">
        <img src="{{ url('img/back.png') }}" alt="Back" width="30" />
    </a>

    <!-- Kontainer Utama -->
    <div class="container-box">

        <!-- Logo Skul.id dan PU -->
        <div class="logo-wrapper d-flex align-items-center gap-2 mb-4">
            <img src="{{ url('img/logo.png') }}" alt="Logo Skul.id" class="logo-skulid" />
            <img src="{{ url('img/pu.png') }}" alt="Logo PU" class="logo-pu" />
        </div>

        <div class="inner-content">

            <!-- Icon besar -->
            <div class="icon-besar">
                <img src="{{ url('img/login-mitra.png') }}" alt="Icon Besar" style="width: 350px;">
            </div>

            <!-- Form Register -->
            <div class="login-form">
                <div class="text">
                    <h1>Daftar Mitra</h1>
                    <p>Sudah punya akun? <a href="{{ route('login.mitra') }}" class="text-danger">Login Sekarang</a></p>
                </div>

                <form action="{{ route('register.store') }}" method="POST">
                    @csrf
                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-phone"></i></span>
                        <input type="text" class="form-control" placeholder="Nomor HP Telkomsel" name="no_hp"
                            required>
                        @if ($errors->has('no_hp'))
                            <div class="alert alert-danger mt-2">
                                {{ $errors->first('no_hp') }}
                            </div>
                        @endif
                    </div>

                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-lock"></i></span>
                        <input type="password" class="form-control" placeholder="Password" name="password"
                            id="password" required>
                        <span class="position-absolute top-50 end-0 translate-middle-y pe-3"
                            onclick="togglePassword('password', 'icon1')" style="cursor: pointer;">
                            <i class="bi bi-eye-slash" id="icon1"></i>
                        </span>
                    </div>

                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
                        <input type="password" class="form-control" placeholder="Konfirmasi Password"
                            name="password_confirmation" id="passwordConfirm" required>
                        <span class="position-absolute top-50 end-0 translate-middle-y pe-3"
                            onclick="togglePassword('passwordConfirm', 'icon2')" style="cursor: pointer;">
                            <i class="bi bi-eye-slash" id="icon2"></i>
                        </span>
                    </div>

                    <input type="hidden" name="role" value="mitra">

                    <div class="form-check mb-3 text-start">
                        <input class="form-check-input" type="checkbox" id="termsCheck" required>
                        <label class="form-check-label" for="termsCheck">
                            Dengan mendaftar, saya setuju dengan <a href="#">Ketentuan Pengguna</a> & <a
                                href="#">Kebijakan Privasi</a>
                        </label>
                    </div>

                    <button type="submit" class="btn login-btn w-100">DAFTAR</button>
                </form>
            </div>
        </div>
    </div>

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

        // Validasi frontend Telkomsel & Password
        document.querySelector("form").addEventListener("submit", function(e) {
            const phoneInput = document.querySelector('input[name="no_hp"]');
            const passwordInput = document.querySelector('input[name="password"]');

            const phonePattern = /^08(11|12|13|21|22|23|51|52|53)[0-9]{5,8}$/;
            const passwordValue = passwordInput.value;
            const phoneValue = phoneInput.value;

            let errorMessages = [];

            if (!phonePattern.test(phoneValue)) {
                errorMessages.push("Nomor HP harus nomor Telkomsel.");
            }

            if (passwordValue.length < 8) {
                errorMessages.push("Password minimal 8 karakter.");
            }

            if (errorMessages.length > 0) {
                e.preventDefault();
                alert(errorMessages.join("\n"));
            }
        });

        // Hapus alert error setelah beberapa detik
        setTimeout(() => {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                alert.classList.add('fade');
                setTimeout(() => alert.remove(), 500);
            });
        }, 2000);
    </script>

</body>

</html>
