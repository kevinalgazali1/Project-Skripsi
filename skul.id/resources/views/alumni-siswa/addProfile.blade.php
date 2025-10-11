<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    {{-- <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" /> --}}
    <link href="https://cdn.jsdelivr.net/npm/tom-select@2.3.0/dist/css/tom-select.css" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1e40af;
            --bg-light: #f9fafb;
            --border: #e5e7eb;
            --text: #1f2937;
            --muted: #6b7280;
            --radius: 12px;
            --input-padding: 12px 16px;
            --transition: 0.3s ease;
        }

        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #fff;
            color: var(--text);
            margin: 0;
        }

        .container-custom {
            display: flex;
            min-height: 100vh;
            width: 100vw;
            align-items: center;
            justify-content: center;
        }

        .card-custom {
            display: flex;
            height: 100vh;
            width: 100%;
            overflow: hidden;
        }

        .card-image {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .card-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .card-form {
            flex: 1;
            background-color: white;
            padding: 60px 80px;
            overflow-y: auto;
            height: 100vh;
        }

        .card-form h2 {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 30px;
            color: var(--primary);
        }

        .form-group,
        .mb-3 {
            margin-bottom: 20px;
            position: relative;
        }

        .form-group label {
            display: block;
            margin-bottom: 6px;
            font-size: 14px;
            font-weight: 600;
            color: var(--muted);
        }

        input[type="text"],
        input[type="email"],
        input[type="number"],
        input[type="date"],
        select,
        textarea {
            width: 100%;
            padding: var(--input-padding);
            border: 1.5px solid var(--border);
            border-radius: var(--radius);
            background-color: var(--bg-light);
            font-size: 15px;
            color: var(--text);
            transition: border var(--transition), box-shadow var(--transition);
        }

        input:focus,
        select:focus,
        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
            outline: none;
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
            transition: background-color var(--transition);
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        .alert {
            font-size: 14px;
        }

        @media (max-width: 768px) {
            .card-custom {
                flex-direction: column;
                height: auto;
            }

            .card-image {
                height: 240px;
            }

            .card-form {
                padding: 30px 20px;
                max-height: none;
            }

            .img-fluid {
                width: 130px;
            }
        }

        #pekerjaan-fields {
            display: none;
        }

        .is-invalid {
            border: 1px solid red !important;
            background-color: #ffe6e6;
        }
    </style>
</head>

<body>
    <div class="container-custom">
        <div class="card-custom">

            <div class="card-image">
                <img src="{{ url('img/edit-profile.png') }}" alt="Edit Profile">
            </div>

            <div class="card-form">
                <h2>Data Diri</h2>

                <form action="{{ route('alumni-siswa.storeProfile') }}" method="POST" enctype="multipart/form-data">
                    @csrf

                    <div class="form-group">
                        <label>Status</label>
                        <input type="text" name="status" value="Alumni" readonly>
                    </div>

                    <div class="form-group">
                        <label>Nama Lengkap</label>
                        <input type="text" name="nama_lengkap" value="{{ old('nama_lengkap') }}"
                            data-required="true">
                    </div>

                    <div class="form-group" id="nik-group">
                        <label>Nomor Induk Kependudukan (NIK)</label>
                        <input type="text" name="nik" value="{{ old('nik') }}" data-required="true">
                    </div>
                    <div class="form-group" id="tahun-lulusan-group">
                        <label>Tahun Kelulusan</label>
                        <input type="number" name="tahun_kelulusan" value="{{ old('tahun_kelulusan') }}"
                            data-required="true">
                    </div>

                    <div class="mb-3">
                        <label class="form-label fw-semibold">Provinsi</label>
                        <select class="form-select rounded-3" id="provinsi-select" name="provinsi" data-required="true">
                            <option selected disabled>Pilih Provinsi</option>
                            @foreach ($provinsi as $prov)
                                <option value="{{ $prov['name'] }}" data-id="{{ $prov['id'] }}">
                                    {{ $prov['name'] }}
                                </option>
                            @endforeach
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label fw-semibold">Kota/Kabupaten</label>
                        <select class="form-select rounded-3" id="kota-select" name="kota" data-required="true">
                            <option selected disabled>Pilih Kota/Kabupaten</option>
                            @foreach ($kabupaten as $kab)
                                <option value="{{ $kab['name'] }}" data-id="{{ $kab['id'] }}">
                                    {{ $kab['name'] }}
                                </option>
                            @endforeach
                        </select>
                    </div>

                    <div class="mb-3">
                        <label for="asal_sekolah">Asal Sekolah (SMA/SMK)</label>
                        <select name="asal_sekolah" id="asal_sekolah" placeholder="Cari nama sekolah..."
                            class="form-select" data-required="true"></select>
                    </div>

                    <div class="mb-3">
                        <label for="npsn">NPSN</label>
                        <input type="text" id="npsn" name="npsn" class="form-control" readonly>
                    </div>



                    <div class="mb-3">
                        <label for="jurusan_sekolah">Jurusan Sekolah</label>
                        <input list="daftar-jurusan" name="jurusan_sekolah" id="jurusan_sekolah" class="form-control"
                            value="{{ old('jurusan_sekolah') }}" data-required="true">
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

                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" name="email" value="{{ old('email') }}" data-required="true">
                    </div>

                    <div class="form-group">
                        <label>No. Telepon</label>
                        <input type="text" name="no_telepon" value="{{ $no_hp }}" readonly>
                    </div>

                    <div class="form-group">
                        <label>Jenis Kelamin</label>
                        <select name="jenis_kelamin" data-required="true">
                            <option value="laki-laki" selected>Laki-laki</option>
                            <option value="perempuan">Perempuan</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Tanggal Lahir</label>
                        <input type="date" name="tanggal_lahir" value="{{ old('tanggal_lahir') }}"
                            data-required="true">
                    </div>

                    <div class="form-group">
                        <label>Status Saat Ini</label>
                        <select name="status_saat_ini" id="status_saat_ini" data-required="true">
                            <option disabled selected>Pilih Status</option>
                            <option value="Bekerja">Bekerja</option>
                            <option value="Wirausaha">Wirausaha</option>
                            <option value="Kuliah">Kuliah</option>
                            <option value="Tidak Bekerja">Tidak Bekerja</option>
                        </select>
                    </div>

                    <!-- Form jika status = Bekerja atau Wirausaha -->
                    <div id="pekerjaan-fields" style="display: none;">
                        <div class="form-group">
                            <label>Sertifikasi Terakhir yang Diikuti</label>
                            <input type="text" name="sertifikasi_terakhir"
                                placeholder="Contoh: Junior Web Developer">
                        </div>

                        <div class="form-group">
                            <label>Apakah Sertifikasi Sesuai dengan Pekerjaan?</label>
                            <select name="kesesuaian_sertifikasi">
                                <option disabled selected>Pilih Jawaban</option>
                                <option>Ya, sesuai</option>
                                <option>Tidak sesuai</option>
                                <option>Tidak relevan / Belum bekerja</option>
                            </select>
                        </div>
                    </div>

                    <!-- Form jika status = Kuliah -->
                    <div id="kuliah-fields" style="display: none;">
                        <div class="form-group">
                            <label>Nama Perguruan Tinggi</label>
                            <input type="text" name="nama_universitas"
                                placeholder="Contoh: Universitas Gadjah Mada">
                        </div>

                        <div class="form-group">
                            <label>Program Studi</label>
                            <input type="text" name="jurusan_universitas"
                                placeholder="Contoh: Teknik Informatika">
                        </div>

                        <div class="form-group">
                            <label>Sertifikasi Terakhir yang Diikuti</label>
                            <input type="text" name="sertifikasi_terakhir"
                                placeholder="Contoh: Junior Web Developer">
                        </div>
                    </div>

                    <!-- Form jika status = Tidak Bekerja -->
                    <div id="nonjob-fields" style="display: none;">
                        <div class="form-group">
                            <label>Sertifikasi Terakhir yang Diikuti</label>
                            <input type="text" name="sertifikasi_terakhir"
                                placeholder="Contoh: Junior Web Developer">
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Bidang Pekerjaan yang diminati</label>
                        <input type="text" name="bidang_pekerjaan" placeholder="Contoh: Teknologi Informasi">
                    </div>

                    <div class="form-group">
                        <label>Skill</label>
                        <input name="skills">{{ old('skills') }}</input>
                    </div>

                    <div class="form-group">
                        <label>Alamat</label>
                        <textarea name="alamat" rows="2">{{ old('alamat') }}</textarea>
                    </div>

                    <div class="mb-3">
                        <label for="foto" class="form-label">Upload Foto
                            Profile</label>
                        <input type="file" class="form-control" id="foto_profil" name="foto_profil"
                            accept=".jpg,.jpeg,.png" data-required="true">
                    </div>

                    <button type="submit">Simpan Perubahan</button>

                    <div class="container my-5 text-center">
                        <p class="mb-4">Our Partner</p>
                </form>
                <div class="d-flex justify-content-center align-items-center gap-4 flex-wrap">
                    <img src="{{ url('img/telkomsel.png') }}" alt="Partner 1" class="img-fluid"
                        style="max-height: 50px;">
                    <img src="{{ url('img/pu.png') }}" alt="Partner 2" class="img-fluid" style="max-height: 50px;">
                </div>
            </div>
        </div>
    </div>
    </div>


    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    @if ($errors->has('email'))
        <script>
            Swal.fire({
                icon: 'error',
                title: 'Email sudah digunakan!',
                text: '{{ $errors->first('email') }}',
                confirmButtonText: 'Oke'
            });
        </script>
    @endif
    <script>
        document.querySelector('form').addEventListener('submit', function(e) {
            const requiredFields = this.querySelectorAll('[data-required]');
            let emptyFields = [];

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    emptyFields.push(field);
                }
            });

            if (emptyFields.length > 0) {
                e.preventDefault();

                // Tambah class 'is-invalid' ke field kosong
                emptyFields.forEach(field => field.classList.add('is-invalid'));

                Swal.fire({
                    icon: 'error',
                    title: 'Data belum lengkap!',
                    text: 'Harap isi semua field yang wajib diisi.',
                    confirmButtonText: 'Oke'
                });

                return false;
            } else {
                // Hapus class 'is-invalid' jika sudah terisi
                requiredFields.forEach(field => field.classList.remove('is-invalid'));
            }
        });

        // Hapus 'is-invalid' saat mengetik
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
            const status = document.getElementById('status-select').value;
            const kelasGroup = document.getElementById('kelas-group');
            const kelasInput = document.querySelector('select[name="kelas"]');
            const tahunGroup = document.getElementById('tahun-lulusan-group');
            const tahunInput = document.querySelector('input[name="tahun_kelulusan"]');

            if (status === 'alumni') {
                kelasGroup.style.display = 'none';
                kelasInput.removeAttribute('required');
                tahunGroup.style.display = 'block';
                tahunInput.setAttribute('required', true);
            } else {
                kelasGroup.style.display = 'block';
                kelasInput.setAttribute('required', true);
                tahunGroup.style.display = 'none';
                tahunInput.removeAttribute('required');
            }
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

            // Reset semua ke hidden
            pekerjaanFields.style.display = 'none';
            kuliahFields.style.display = 'none';
            nonjobFields.style.display = 'none';

            // Tampilkan sesuai status
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
            const event = new Event('change');
            document.getElementById('status_saat_ini').dispatchEvent(event);
        });
    </script>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
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
