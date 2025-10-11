<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/tom-select@2.3.0/dist/css/tom-select.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            background-color: #fff;
            overflow: hidden;
        }

        a {
            text-decoration: none;
        }

        .container-fluid {
            padding: 0;
            margin: 0;
        }

        /* ============ Navbar ============ */
        .navbar {
            background-color: #ffffff;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navbar .logo {
            width: 80px;
            margin-left: 30px;
        }

        .user-info {
            display: flex;
            align-items: center;
            margin-right: 30px;
        }

        .user-info span {
            margin-right: 1rem;
            font-weight: 600;
        }

        .user-info .profile-picture {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #c04055;
        }

        /* ============ Layout ============ */
        .main-wrapper {
            display: flex;
            height: calc(100vh - 80px);
        }

        .sidebar {
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
            height: 100%;
            background-color: #eff9ff;
            position: relative;
            flex-shrink: 0;
        }

        .nav-link {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            color: #495057;
            text-decoration: none;
            transition: background 0.2s, color 0.2s;
        }

        .nav-link.active {
            background-color: #e7f1ff;
            color: #0d6efd;
            font-weight: 600;
        }

        .nav-link i {
            font-size: 1.1rem;
        }

        .sidebar h5 {
            letter-spacing: 0.5px;
        }

        .logout {
            cursor: pointer;
        }

        .content {
            flex: 1;
            overflow-y: auto;
            background-color: #fff;
        }


        /* ============ Responsive ============ */
        @media (max-width: 768px) {
            .main-wrapper {
                flex-direction: column;
            }

            .sidebar {
                display: none;
            }

            .user-info {
                display: none;
                ;
            }

            .banner-images {
                flex-direction: column;
                align-items: flex-start;
            }

            .banner-images img {
                width: 100%;
            }

            .banner .illustration {
                display: none;
            }
        }

        .partner-footer-logo {
            height: 40px;
            object-fit: contain;
        }

        .footer ul {
            padding-left: 0;
        }

        .footer li {
            margin-bottom: 8px;
        }

        .footer .bg-opacity-50 {
            background-color: rgba(255, 255, 255, 0.75) !important;
            /* semi-transparent white for readability */
        }

        .container-footer {
            width: 100%;
            height: 100%;
            ;
        }
    </style>
</head>

<body>
    <div class="navbar">
        <div class="col-lg-6 d-flex">
            <div class="col-lg-2">
                <img src="{{ url('img/logo.png') }}" alt="Logo" class="logo" width="" />
            </div>
        </div>
        <div class="user-info">
            <span>Halo, {{ $user->alumniSiswaProfile->nama_lengkap }}</span>
            <img src="{{ asset('storage/' . $user->alumniSiswaProfile->foto_profil) }}" alt="Profile"
                class="profile-picture" />
        </div>
        <button class="btn d-md-none" type="button" data-bs-toggle="offcanvas" data-bs-target="#offcanvasSidebar">
            <i class="bi bi-list fs-3"></i>
        </button>
    </div>

    <div class="offcanvas offcanvas-start" tabindex="-1" id="offcanvasSidebar">
        <div class="offcanvas-header">
            <h5 class="offcanvas-title">Menu</h5>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas"></button>
        </div>
        <div class="offcanvas-body">
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.index') }}"><i
                    class="bi bi-house-door-fill me-2"></i>Beranda</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.sertifikasi') }}"><i
                    class="bi bi-patch-check-fill me-2"></i>Sertifikasi</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.loker') }}"><i
                    class="bi bi-briefcase-fill me-2"></i>Loker</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.pelatihan') }}"><i
                    class="bi bi-journal-text me-2"></i>Pelatihan</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.profile') }}"><i
                    class="bi bi-person-fill me-2"></i>Profil</a>
            <form id="logout-form" action="{{ route('logout') }}" method="POST">
                @csrf
                <a href="#" class="logout" onclick="event.preventDefault(); confirmLogout();">
                    <i class="bi bi-box-arrow-right me-2"></i>Logout</a>
            </form>
        </div>
    </div>

    <div class="main-wrapper">
        <div class="sidebar sidebar-main d-flex flex-column p-3" style="width: 250px;">


            <!-- Navigation Links -->
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.index') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.index') }}">
                <i class="bi bi-house-door-fill me-2"></i> Beranda
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.sertifikasi') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.sertifikasi') }}">
                <i class="bi bi-patch-check-fill me-2"></i> Sertifikasi
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.loker') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.loker') }}">
                <i class="bi bi-briefcase-fill me-2"></i> Loker
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.pelatihan') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.pelatihan') }}">
                <i class="bi bi-journal-text me-2"></i> Pelatihan
            </a>
            <a class="nav-link d-flex align-items-center mb-4 {{ request()->routeIs('alumni-siswa.profile') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.profile') }}">
                <i class="bi bi-person-fill me-2"></i> Profil
            </a>

            <!-- Logout -->
            <form id="logout-form" action="{{ route('logout') }}" method="POST" class="mt-auto">
                @csrf
                <a href="#" class="nav-link d-flex align-items-center text-danger fw-semibold mb-4"
                    onclick="event.preventDefault(); confirmLogout();">
                    <i class="bi bi-box-arrow-right me-2"></i> Logout
                </a>
            </form>
        </div>

        <!-- Content -->
        <div class="content" id="edit">
            <div class="container py-5 px-4">
                <div class="card rounded-4 p-4" style="background-color: #fefefe;">
                    <h3 class="mb-4 fw-bold text-primary text-center">Edit Profil</h3>
                    <form action="{{ route('alumni-siswa.updateProfile') }}" method="POST" id="form-edit-profil"
                        enctype="multipart/form-data">
                        @csrf
                        @method('PUT')

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Nama Lengkap</label>
                            <input type="text" name="nama_lengkap" class="form-control rounded-3"
                                value="{{ old('nama_lengkap', $user->alumniSiswaProfile->nama_lengkap ?? '') }}"
                                data-required="true">
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Nomor Induk Kependudukan (NIK)</label>
                            <input type="text" name="nik" class="form-control rounded-3"
                                value="{{ old('nik', $user->alumniSiswaProfile->nik ?? '') }}" data-required="true">
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Tahun Kelulusan</label>
                            <input type="number" name="tahun_kelulusan" class="form-control rounded-3"
                                value="{{ old('tahun_kelulusan', $user->alumniSiswaProfile->tahun_kelulusan ?? '') }}"
                                data-required="true">
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Provinsi</label>
                            <select class="form-select rounded-3" id="provinsi-select" name="provinsi"
                                data-required="true">
                                <option selected disabled>Pilih Provinsi</option>
                                @foreach ($provinsi as $prov)
                                    <option value="{{ $prov['name'] }}"
                                        {{ old('provinsi', $user->alumniSiswaProfile->provinsi ?? '') == $prov['name'] ? 'selected' : '' }}
                                        data-id="{{ $prov['id'] }}">
                                        {{ $prov['name'] }}
                                    </option>
                                @endforeach
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Kota/Kabupaten</label>
                            <select class="form-select rounded-3" id="kota-select" name="kota"
                                data-required="true">
                                <option selected disabled>Pilih Kota/Kabupaten</option>
                                @foreach ($kabupaten as $kab)
                                    <option value="{{ $kab['name'] }}"
                                        {{ old('kota', $user->alumniSiswaProfile->kota ?? '') == $kab['name'] ? 'selected' : '' }}
                                        data-id="{{ $kab['id'] }}">
                                        {{ $kab['name'] }}
                                    </option>
                                @endforeach
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold" for="asal_sekolah">Asal Sekolah (SMA/SMK)</label>
                            <select name="asal_sekolah" id="asal_sekolah" class="form-select" data-required="true">
                                @php
                                    $asalSekolah = old('asal_sekolah', $user->alumniSiswaProfile->asal_sekolah ?? '');
                                @endphp
                                @if ($asalSekolah)
                                    <option value="{{ $asalSekolah }}" selected>{{ $asalSekolah }}</option>
                                @endif
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold" for="npsn">NPSN</label>
                            <input type="text" id="npsn" name="npsn" class="form-control rounded-3"
                                readonly value="{{ old('npsn', $user->alumniSiswaProfile->npsn ?? '') }}">
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold" for="jurusan_sekolah">Jurusan Sekolah</label>
                            <input list="daftar-jurusan" name="jurusan_sekolah" id="jurusan_sekolah"
                                class="form-control rounded-3"
                                value="{{ old('jurusan_sekolah', $user->alumniSiswaProfile->jurusan_sekolah) }}"
                                data-required="true">
                            <datalist id="daftar-jurusan">
                                <option value="Desain Pemodelan dan Informasi Bangunan (DPIB)">
                                <option value="Bisnis Konstruksi dan Properti (BKP)">
                                <option value="Teknik Konstruksi dan Perumahan (TKP)">
                                <option value="Teknik Geomatika">
                                <option value="Rekayasa Perangkat Lunak (RPL)">
                                <option value="Teknik Komputer dan Jaringan (TKJ)">
                                <option value="Teknik Otomotif">
                                <option value="Teknik Elektronika Industri">
                                <option value="Teknik Pengelasan">
                            </datalist>
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Email</label>
                            <input type="email" name="email" class="form-control rounded-3"
                                value="{{ old('email', $user->email) }}" data-required="true">
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">No. Telepon</label>
                            <input type="text" name="no_hp" class="form-control rounded-3"
                                value="{{ $user->no_hp }}" readonly>
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Jenis Kelamin</label>
                            <select name="jenis_kelamin" class="form-select rounded-3" data-required="true">
                                <option value="laki-laki"
                                    {{ old('jenis_kelamin', $user->alumniSiswaProfile->jenis_kelamin ?? '') == 'laki-laki' ? 'selected' : '' }}>
                                    Laki-laki</option>
                                <option value="perempuan"
                                    {{ old('jenis_kelamin', $user->alumniSiswaProfile->jenis_kelamin ?? '') == 'perempuan' ? 'selected' : '' }}>
                                    Perempuan</option>
                            </select>
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Tanggal Lahir</label>
                            <input type="date" name="tanggal_lahir" class="form-control rounded-3"
                                value="{{ old('tanggal_lahir', $user->alumniSiswaProfile->tanggal_lahir) }}"
                                data-required="true">
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Status Saat Ini</label>
                            <select name="status_saat_ini" class="form-select rounded-3" id="status_saat_ini"
                                data-required="true">
                                <option disabled
                                    {{ old('status_saat_ini', $user->alumniSiswaProfile->status_saat_ini ?? '') == '' ? 'selected' : '' }}>
                                    Pilih Status</option>
                                <option value="Bekerja"
                                    {{ old('status_saat_ini', $user->alumniSiswaProfile->status_saat_ini ?? '') == 'Bekerja' ? 'selected' : '' }}>
                                    Bekerja</option>
                                <option value="Wirausaha"
                                    {{ old('status_saat_ini', $user->alumniSiswaProfile->status_saat_ini ?? '') == 'Wirausaha' ? 'selected' : '' }}>
                                    Wirausaha</option>
                                <option value="Kuliah"
                                    {{ old('status_saat_ini', $user->alumniSiswaProfile->status_saat_ini ?? '') == 'Kuliah' ? 'selected' : '' }}>
                                    Kuliah</option>
                                <option value="Tidak Bekerja"
                                    {{ old('status_saat_ini', $user->alumniSiswaProfile->status_saat_ini ?? '') == 'Tidak Bekerja' ? 'selected' : '' }}>
                                    Tidak Bekerja</option>
                            </select>
                        </div>

                        <!-- Form jika status = Bekerja atau Wirausaha -->
                        <div id="pekerjaan-fields" style="display: none;">
                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Sertifikasi Terakhir yang Diikuti</label>
                                <input type="text" name="sertifikasi_terakhir" class="form-control rounded-3"
                                    value="{{ old('sertifikasi_terakhir', $user->alumniSiswaProfile->sertifikasi_terakhir) }}"
                                    placeholder="Contoh: Junior Web Developer">
                            </div>

                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Apakah Sertifikasi Sesuai dengan
                                    Pekerjaan?</label>
                                <select name="kesesuaian_sertifikasi" class="form-select rounded-3">
                                    <option disabled
                                        {{ old('kesesuaian_sertifikasi', $user->alumniSiswaProfile->kesesuaian_sertifikasi ?? '') == '' ? 'selected' : '' }}>
                                        Pilih Jawaban
                                    </option>
                                    <option value="Ya, sesuai"
                                        {{ old('kesesuaian_sertifikasi', $user->alumniSiswaProfile->kesesuaian_sertifikasi ?? '') == 'Ya, sesuai' ? 'selected' : '' }}>
                                        Ya, sesuai
                                    </option>
                                    <option value="Tidak sesuai"
                                        {{ old('kesesuaian_sertifikasi', $user->alumniSiswaProfile->kesesuaian_sertifikasi ?? '') == 'Tidak sesuai' ? 'selected' : '' }}>
                                        Tidak sesuai
                                    </option>
                                    <option value="Tidak relevan / Belum bekerja"
                                        {{ old('kesesuaian_sertifikasi', $user->alumniSiswaProfile->kesesuaian_sertifikasi ?? '') == 'Tidak relevan / Belum bekerja' ? 'selected' : '' }}>
                                        Tidak relevan / Belum bekerja
                                    </option>
                                </select>
                            </div>
                        </div>

                        <!-- Form jika status = Kuliah -->
                        <div id="kuliah-fields" style="display: none;">
                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Nama Perguruan Tinggi</label>
                                <input type="text" name="nama_universitas" class="form-control rounded-3"
                                    value="{{ old('nama_universitas', $user->alumniSiswaProfile->nama_universitas) }}"
                                    placeholder="Contoh: Universitas Gadjah Mada">
                            </div>

                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Program Studi</label>
                                <input type="text" name="jurusan_universitas" class="form-control rounded-3"
                                    value="{{ old('jurusan_universitas', $user->alumniSiswaProfile->jurusan_universitas) }}"
                                    placeholder="Contoh: Teknik Informatika">
                            </div>

                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Sertifikasi Terakhir yang Diikuti</label>
                                <input type="text" name="sertifikasi_terakhir" class="form-control rounded-3"
                                    value="{{ old('sertifikasi_terakhir', $user->alumniSiswaProfile->sertifikasi_terakhir) }}"
                                    placeholder="Contoh: Junior Web Developer">
                            </div>
                        </div>

                        <!-- Form jika status = Tidak Bekerja -->
                        <div id="nonjob-fields" style="display: none;">
                            <div class="form-group mb-3">
                                <label class="form-label fw-semibold">Sertifikasi Terakhir yang Diikuti</label>
                                <input type="text" name="sertifikasi_terakhir" class="form-control rounded-3"
                                    value="{{ old('sertifikasi_terakhir', $user->alumniSiswaProfile->sertifikasi_terakhir) }}"
                                    placeholder="Contoh: Junior Web Developer">
                            </div>
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Bidang Pekerjaan yang diminati</label>
                            <input type="text" name="bidang_pekerjaan" class="form-control rounded-3"
                                value="{{ old('bidang_pekerjaan', $user->alumniSiswaProfile->bidang_pekerjaan) }}"
                                placeholder="Contoh: Teknologi Informasi">
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Skill</label>
                            <input type="text" name="skills"
                                class="form-control rounded-3" value="{{ old('skills', $user->alumniSiswaProfile->skills ?? '') }}">
                        </div>

                        <div class="form-group mb-3">
                            <label class="form-label fw-semibold">Alamat</label>
                            <textarea name="alamat" class="form-control rounded-3" rows="2">{{ old('alamat', $user->alumniSiswaProfile->alamat ?? '') }}</textarea>
                        </div>

                        <div class="mb-3">
                            <label for="edit_foto_profil" class="form-label">Upload
                                Foto
                                Baru
                                (Opsional)</label>
                            <input type="file" class="form-control" id="edit_foto_profil" name="foto_profil"
                                accept=".jpg,.jpeg,.png">
                            <img id="preview_foto_edit" class="img-fluid mt-2 rounded" style="max-height: 150px;">
                        </div>

                        <div class="text-end">
                            <button type="submit" class="btn btn-success px-4 py-2 rounded-3 fw-semibold">Simpan
                                Perubahan</button>
                        </div>
                    </form>

                </div>
            </div>
            <footer class="footer text-dark"
                style="background: url('{{ url('img/footer.png') }}') no-repeat center center / cover;">
                <div class="container-footer">
                    <div class="row align-items-start bg-white bg-opacity-75 rounded-3 p-4 shadow-sm">
                        <!-- Logo & Deskripsi -->
                        <div class="col-md-4 mb-4">
                            <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id" style="height: 65px;"
                                class="mb-3">
                            <p class="small">
                                mitraskul.Id adalah platform yang menghubungkan alumni dengan sekolah, dunia industri,
                                dan
                                peluang karier.
                            </p>
                        </div>

                        <!-- Kontak -->
                        <div class="col-md-4 mb-4">
                            <!-- <h6 class="fw-bold text-uppercase mb-3">Kontak</h6>
                            <ul class="list-unstyled small">
                                <li><i class="bi bi-envelope-fill me-2"></i>mitraskulid@gmail.com</li>
                                <li><i class="bi bi-telephone-fill me-2"></i>+62 851-7959-2408</li>
                            </ul> -->
                        </div>

                        <!-- Partner -->
                        <div class="col-md-4 mb-4">
                            <h6 class="fw-bold text-uppercase mb-3">Didukung Oleh</h6>
                            <div class="d-flex align-items-center gap-3 flex-wrap">
                                <img src="{{ url('img/Telkomsel.png') }}" alt="Telkomsel"
                                    class="partner-footer-logo">
                                <img src="{{ url('img/pu.png') }}" alt="PUP Makassar" class="partner-footer-logo">
                            </div>
                        </div>
                        <div class="text-center text-dark small mt-4">
                            © 2025 mitraskul.Id. All rights reserved.
                        </div>
                    </div>
                </div>
            </footer>
        </div>


        <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

        @if ($errors->any())
            <script>
                Swal.fire({
                    icon: 'error',
                    title: 'Terjadi Kesalahan!',
                    html: `{!! implode('<br>', $errors->all()) !!}`,
                    confirmButtonText: 'Oke'
                });
            </script>
        @endif
        <script>
            function confirmLogout() {
                event.preventDefault();
                Swal.fire({
                    title: 'Yakin ingin logout?',
                    text: "Anda akan keluar dari akun ini.",
                    icon: 'warning',
                    showCancelButton: true,
                    confirmButtonColor: '#d33',
                    cancelButtonColor: '#6c757d',
                    confirmButtonText: 'Ya, logout!',
                    cancelButtonText: 'Batal'
                }).then((result) => {
                    if (result.isConfirmed) {
                        document.getElementById('logout-form').submit();
                    }
                });
            }
        </script>

        <script>
            document.getElementById('form-edit-profil').addEventListener('submit', function(e) {
                const requiredFields = this.querySelectorAll('[data-required]');
                let emptyFields = [];

                requiredFields.forEach(field => {
                    if (!field.value.trim()) {
                        emptyFields.push(field);
                    }
                });

                if (emptyFields.length > 0) {
                    e.preventDefault();

                    emptyFields.forEach(field => field.classList.add('is-invalid'));

                    Swal.fire({
                        icon: 'error',
                        title: 'Data belum lengkap!',
                        text: 'Harap isi semua field yang wajib diisi.',
                        confirmButtonText: 'Oke'
                    });

                    return false;
                } else {
                    requiredFields.forEach(field => field.classList.remove('is-invalid'));
                }
            });

            // Hapus class saat diketik
            document.querySelectorAll('[data-required]').forEach(function(field) {
                field.addEventListener('input', function() {
                    if (this.value.trim() !== '') {
                        this.classList.remove('is-invalid');
                    }
                });
            });
        </script>

        <script>
            document.getElementById('provinsi-select').addEventListener('change', function() {
                const provId = this.options[this.selectedIndex].getAttribute('data-id');

                fetch(`https://www.emsifa.com/api-wilayah-indonesia/api/regencies/${provId}.json`)
                    .then(response => response.json())
                    .then(data => {
                        const kotaSelect = document.getElementById('kota-select');
                        kotaSelect.innerHTML = '<option selected disabled>Pilih Kota/Kabupaten</option>';

                        data.forEach(kota => {
                            const opt = document.createElement('option');
                            opt.value = kota.name;
                            opt.textContent = kota.name;
                            kotaSelect.appendChild(opt);
                        });
                    });
            });

            function toggleStatus() {
                const kelasGroup = document.getElementById('kelas-group');
                const kelasInput = document.querySelector('select[name="kelas"]');
                const tahunGroup = document.getElementById('tahun-lulusan-group');
                const tahunInput = document.querySelector('input[name="tahun_kelulusan"]');

            }

            document.addEventListener('DOMContentLoaded', toggleStatus);

            setTimeout(() => {
                document.getElementById('success-message')?.remove();
                document.getElementById('error-message')?.remove();
            }, 5000);
            // Success and error fade
            setTimeout(() => {
                const msg = document.getElementById('success-message');
                if (msg) {
                    msg.style.transition = 'opacity 0.5s ease-out';
                    msg.style.opacity = '0';
                    setTimeout(() => msg.remove(), 500);
                }

                const errorMsg = document.getElementById('error-message');
                if (errorMsg) {
                    errorMsg.style.transition = 'opacity 0.5s ease-out';
                    errorMsg.style.opacity = '0';
                    setTimeout(() => errorMsg.remove(), 500);
                }
            }, 5000);

            function smoothScrollTo(targetY, duration) {
                const startY = window.scrollY;
                const distance = targetY - startY;
                let startTime = null;

                function animation(currentTime) {
                    if (!startTime) startTime = currentTime;
                    const timeElapsed = currentTime - startTime;
                    const progress = Math.min(timeElapsed / duration, 1);
                    window.scrollTo(0, startY + distance * easeInOutQuad(progress));
                    if (progress < 1) {
                        requestAnimationFrame(animation);
                    }
                }

                function easeInOutQuad(t) {
                    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                }

                requestAnimationFrame(animation);
            }

            window.addEventListener('load', function() {
                if (window.innerWidth <= 768) {
                    setTimeout(function() {
                        const targetElement = document.querySelector('.card-form');
                        const targetY = targetElement.getBoundingClientRect().top + window.scrollY;
                        smoothScrollTo(targetY, 1500); // 1500ms = 1.5 detik durasi scroll
                    }, 1500); // jeda sebelum mulai scroll
                }
            });

            document.getElementById('status_saat_ini').addEventListener('change', function() {
                const status = this.value;
                const pekerjaanFields = document.getElementById('pekerjaan-fields');
                const kuliahFields = document.getElementById('kuliah-fields');
                const nonjobFields = document.getElementById('nonjob-fields');

                pekerjaanFields.style.display = 'none';
                kuliahFields.style.display = 'none';
                nonjobFields.style.display = 'none';

                if (status === 'Bekerja' || status === 'Wirausaha') {
                    pekerjaanFields.style.display = 'block';
                } else if (status === 'Kuliah') {
                    kuliahFields.style.display = 'block';
                } else if (status === 'Tidak Bekerja') {
                    nonjobFields.style.display = 'block';
                }
            });

            // Untuk mempertahankan tampilan saat halaman dimuat ulang
            document.addEventListener('DOMContentLoaded', function() {
                const select = document.getElementById('status_saat_ini');
                if (select) {
                    const event = new Event('change');
                    select.dispatchEvent(event);
                }
            });
        </script>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/tom-select@2.3.0/dist/js/tom-select.complete.min.js"></script>
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                new TomSelect("#asal_sekolah", {
                    valueField: 'text', // <--- simpan nama sekolah
                    labelField: 'text',
                    searchField: 'text',
                    placeholder: 'Cari nama sekolah...',
                    load: function(query, callback) {
                        if (query.length < 3) return callback();

                        fetch(`/alumni-siswa/cari-sekolah?q=${encodeURIComponent(query)}`)
                            .then(res => res.json())
                            .then(data => {
                                callback(data.results);
                            }).catch(() => {
                                callback();
                            });
                    },
                    onChange: function(value) {
                        // jika kamu masih ingin simpan NPSN, buat input hidden untuk itu
                        const selectedOption = this.options[value];
                        if (selectedOption && selectedOption.id) {
                            document.getElementById('npsn').value = selectedOption.id;
                        }
                    }
                });
            });
        </script>
</body>

</html>
